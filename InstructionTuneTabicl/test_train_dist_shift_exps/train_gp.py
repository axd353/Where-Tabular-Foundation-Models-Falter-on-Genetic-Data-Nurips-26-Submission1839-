"""
train_gp.py
===========
Main entry-point for fine-tuning TabICL (stage-3 strategy) on the
GP-simulation distribution-shift datasets.

Architecture frozen as in Stage-3: col_embedder + row_interactor frozen,
only tf_icl (ICLearning transformer) is trainable.

Usage example (see run_gp.sh):
    python train_gp.py \\
        --data_dir /path/to/data \\
        --checkpoint_path /path/to/stage2.ckpt \\
        --checkpoint_dir /path/to/checkpoints \\
        --wandb_dir /path/to/wandb \\
        --max_steps 100 \\
        --batch_size 8 \\
        --micro_batch_size 1 \\
        --lr 2e-6 \\
        --save_perm_every 10
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Resolve Python path so that tabicl package and local modules are importable
# ---------------------------------------------------------------------------

def _setup_import_paths() -> None:
    this_file = Path(__file__).resolve()
    # tabicl/test_train_dist_shift_exps/train_gp.py  →  tabicl/src
    repo_root = this_file.parents[1]
    src_dir   = repo_root / "src"
    exp_dir   = this_file.parent         # test_train_dist_shift_exps/

    for p in [str(src_dir), str(exp_dir)]:
        if p not in sys.path:
            sys.path.insert(0, p)


_setup_import_paths()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune TabICL (stage-3) on GP-simulation datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Required / key paths -----------------------------------------------
    p.add_argument(
        "--data_dir", required=True,
        help="Root directory containing train_manifest.csv, eval_manifest.csv, and pickles.",
    )
    p.add_argument(
        "--checkpoint_path", default=None,
        help=(
            "Path to a TabICL checkpoint to initialise from (.ckpt).  "
            "If omitted, falls back to the publicly released TabICL classifier."
        ),
    )
    p.add_argument(
        "--checkpoint_dir", required=True,
        help="Parent directory for checkpoints.  A datetime sub-directory is created inside.",
    )
    p.add_argument(
        "--wandb_dir", required=True,
        help="Parent directory for wandb logs.  A datetime sub-directory is created inside.",
    )

    # ---- Training hyper-parameters ------------------------------------------
    p.add_argument("--max_steps",         type=int,   default=100)
    p.add_argument("--batch_size",        type=int,   default=8,
                   help="Number of manifest rows per gradient-update step.")
    p.add_argument("--micro_batch_size",  type=int,   default=1,
                   help="Micro-batch size for gradient accumulation (1 = no accumulation).")
    p.add_argument("--lr",                type=float, default=2e-6)
    p.add_argument("--weight_decay",        type=float, default=0.0,
                   help="AdamW weight decay (L2 regularisation penalty). Default 0 = no regularisation.")
    p.add_argument("--proximal_wd",         action="store_true", default=False,
                   help=(
                       "When set, weight decay regularises toward the initial checkpoint weights "
                       "(proximal/anchor regularisation) instead of toward zero.  "
                       "AdamW's built-in L2 penalty is disabled and replaced by "
                       "(weight_decay/2)*||theta-theta_0||^2 added to the CE loss."
                   ))
    p.add_argument("--scheduler",          type=str,   default="cosine_warmup",
                   help="LR scheduler: constant | cosine_warmup | cosine_with_restarts | linear_warmup | polynomial_decay_warmup.")
    p.add_argument("--warmup_proportion",  type=float, default=0.1,
                   help="Fraction of max_steps used for linear warmup (0.0 = no warmup). -1 = use --warmup_steps instead.")
    p.add_argument("--save_perm_every",   type=int,   default=10,
                   help="Save a permanent checkpoint every N steps.")
    p.add_argument("--save_temp_every",   type=int,   default=1,
                   help="Save a temporary checkpoint every N steps.")

    # ---- Eval settings -------------------------------------------------------
    p.add_argument("--n_eval",        type=int, default=10,
                   help="Number of rows from eval_manifest.csv to evaluate each step.")
    p.add_argument("--eval_seed",     type=int, default=0,
                   help="Seed for sampling the fixed eval batch.")
    p.add_argument("--train_manifest", default="train_manifest.csv",
                   help="Filename of train manifest inside data_dir.")
    p.add_argument("--eval_manifest",  default="eval_manifest.csv",
                   help="Filename of eval manifest inside data_dir.")

    # ---- Wandb settings ------------------------------------------------------
    p.add_argument("--wandb_log",     action="store_true", default=False)
    p.add_argument("--wandb_project", default="TabICL")
    p.add_argument("--wandb_name",    default="GP_stage3")
    p.add_argument("--wandb_mode",    default="offline",
                   choices=["online", "offline", "disabled"])

    # ---- Device / reproducibility -------------------------------------------
    p.add_argument("--device",    default="cuda" if _cuda_available() else "cpu")
    p.add_argument("--np_seed",   type=int, default=42)
    p.add_argument("--torch_seed", type=int, default=42)

    # ---- GP dataset options -------------------------------------------------
    p.add_argument("--max_features",   type=int,  default=None,
                   help="Max features across all datasets (auto-detected if omitted).")
    p.add_argument("--no_cache_pickles", action="store_true", default=False,
                   help="Disable in-memory pickle caching (reduces RAM usage).")

    return p


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _resolve_checkpoint(user_path: str | None) -> str:
    """Return a valid checkpoint path (local file or HuggingFace download)."""
    if user_path:
        ckpt = Path(user_path).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return str(ckpt)

    # Try well-known cached names first
    preferred = [
        "tabicl-classifier-v1-0208.ckpt",
        "tabicl-classifier-v1.1-0506.ckpt",
        "tabicl-classifier.ckpt",
    ]
    hf_cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--jingang--TabICL-clf" / "snapshots"
    if hf_cache_root.exists():
        for name in preferred:
            matches = sorted(hf_cache_root.rglob(name))
            if matches:
                return str(matches[0])

    try:
        from huggingface_hub import hf_hub_download
        for filename in preferred:
            try:
                return hf_hub_download(repo_id="jingang/TabICL-clf", filename=filename)
            except Exception:
                continue
    except Exception:
        pass

    raise FileNotFoundError(
        "No checkpoint found.  Pass --checkpoint_path explicitly or ensure the "
        "TabICL public checkpoint is accessible from jingang/TabICL-clf."
    )


# ---------------------------------------------------------------------------
# Build a tabicl-compatible config namespace
# ---------------------------------------------------------------------------

def _build_tabicl_config(args: argparse.Namespace, ckpt_dir: str, wandb_dir: str) -> argparse.Namespace:
    """
    Construct the config namespace expected by Trainer using tabicl's own
    build_parser().  Stage-3 model architecture defaults are used throughout.
    Extra GP-specific fields are added directly as attributes.
    """
    from tabicl.train.train_config import build_parser

    tabicl_parser = build_parser()

    cli_args = [
        # wandb
        "--wandb_log",     str(args.wandb_log),
        "--wandb_project", args.wandb_project,
        "--wandb_name",    args.wandb_name,
        "--wandb_dir",     wandb_dir,
        "--wandb_mode",    args.wandb_mode,
        # training
        "--device",           args.device,
        "--dtype",            "float32",
        "--np_seed",          str(args.np_seed),
        "--torch_seed",       str(args.torch_seed),
        "--max_steps",        str(args.max_steps),
        "--batch_size",       str(args.batch_size),
        "--micro_batch_size", str(args.micro_batch_size),
        "--lr",               str(args.lr),
        "--weight_decay",     str(args.weight_decay),
        "--scheduler",        args.scheduler,
        "--warmup_proportion", str(args.warmup_proportion),
        "--gradient_clipping","1.0",
        # prior (placeholder values – overridden by GPTrainer.configure_prior)
        "--prior_type",   "mlp_scm",
        "--prior_device", "cpu",
        # model architecture (Stage-3 defaults)
        "--embed_dim",      "128",
        "--col_num_blocks", "3",
        "--col_nhead",      "4",
        "--col_num_inds",   "128",
        "--freeze_col",     "True",
        "--row_num_blocks", "3",
        "--row_nhead",      "8",
        "--row_num_cls",    "4",
        "--row_rope_base",  "100000",
        "--freeze_row",     "True",
        "--icl_num_blocks", "12",
        "--icl_nhead",      "4",
        "--ff_factor",      "2",
        "--norm_first",     "True",
        # checkpointing
        "--checkpoint_dir",   ckpt_dir,
        "--checkpoint_path",  _resolve_checkpoint(args.checkpoint_path),
        "--save_temp_every",  str(args.save_temp_every),
        "--save_perm_every",  str(args.save_perm_every),
        "--only_load_model",  "True",     # load weights only, reset training state
    ]

    config = tabicl_parser.parse_args(cli_args)

    # ---- GP-specific fields (not in tabicl's own parser) --------------------
    config.gp_data_dir       = args.data_dir
    config.gp_train_manifest = args.train_manifest
    config.gp_eval_manifest  = args.eval_manifest
    config.gp_n_eval         = args.n_eval
    config.gp_eval_seed      = args.eval_seed
    config.gp_max_features   = args.max_features   # None → auto-detected
    config.gp_cache_pickles  = not args.no_cache_pickles
    config.gp_proximal_wd    = getattr(args, "proximal_wd", False)

    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # ---- Create datetime sub-directories for outputs -----------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir  = os.path.join(args.checkpoint_dir, run_id)
    wandb_dir = os.path.join(args.wandb_dir,      run_id)
    os.makedirs(ckpt_dir,  exist_ok=True)
    os.makedirs(wandb_dir, exist_ok=True)

    print(f"Run ID         : {run_id}")
    print(f"Checkpoint dir : {ckpt_dir}")
    print(f"Wandb dir      : {wandb_dir}")
    print(f"Data dir       : {args.data_dir}")

    # ---- Set multiprocessing start method ----------------------------------
    from torch.multiprocessing import set_start_method
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass  # already set

    # ---- Build config and run trainer --------------------------------------
    config = _build_tabicl_config(args, ckpt_dir=ckpt_dir, wandb_dir=wandb_dir)

    from gp_trainer import GPTrainer
    trainer = GPTrainer(config)

    print(f"\nStarting training: max_steps={args.max_steps}, "
          f"batch_size={args.batch_size}, lr={args.lr}")
    trainer.train()

    print("\nTraining complete.")
    print(f"Best eval checkpoint: {os.path.join(ckpt_dir, 'best-eval.ckpt')}")


if __name__ == "__main__":
    main()
