#!/usr/bin/env bash
# run_gp_hpo.sh
# -------------
# Launch GP Stage-3 HPO sweeps using train_gp_hpo.py.
#
# Defaults are configured to sweep only learning rate.
# You can later extend to max_steps/batch_size/weight_decay by editing
# the *_VALUES variables below (comma-separated lists).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/train_gp_hpo.py"

# =============================================================================
# USER CONFIG
# =============================================================================

DATA_DIR="../SmalltaskSuite/"
CHECKPOINT_DIR="../../Runs/Checkpoints/"
WANDB_DIR="../Runs/Logs/"
INIT_CHECKPOINT=""

# ---- HPO search space -------------------------------------------------------
# Comma-separated values. Initially vary only LR.
LR_VALUES="1e-7,5e-7,9e-7,1e-6,2e-6,2.5e-6,4.5e-6,6e-6,8e-6,1e-5"
MAX_STEPS_VALUES="500"
BATCH_SIZE_VALUES="10"
WEIGHT_DECAY_VALUES="1e-2,1e-1"

# ---- Proximal weight decay --------------------------------------------------
# Comma-separated list of true/false values to sweep.
#   false       → standard AdamW decay-to-zero  (default)
#   true        → proximal/anchor regularisation toward the initial checkpoint
#   false,true  → sweep both modes (doubles the grid size)
PROXIMAL_WD_VALUES="true"

# ---- Fixed training options -------------------------------------------------
MICRO_BATCH_SIZE=1
SAVE_PERM_EVERY=5
SAVE_TEMP_EVERY=1
N_EVAL=30
EVAL_SEED=0
#
# NUM_ADDITIONAL_CHECKPOINTS – controls how many extra periodic checkpoints are
#   evaluated alongside the best-eval checkpoint during Stage-2 post-hoc eval.
#
#   0  (default) = evaluate only the single best-eval checkpoint saved by
#                  GPTrainer (cheapest; original behaviour).
#
#   N > 0        = also evaluate the N periodic step-*.ckpt checkpoints whose
#                  step number is closest to the best checkpoint's step.  The
#                  final stage2_eval_acc / stage2_eval_ce stored in the HPO
#                  manifest is the MEDIAN across all (N+1) checkpoints.
#                  This reduces the influence of checkpoint-to-checkpoint
#                  training variance on the HPO ranking, at the cost of
#                  N extra forward passes over the Stage-2 eval set per run.
#                  Recommended: 2–4 for a good noise/cost trade-off.
#
NUM_ADDITIONAL_CHECKPOINTS=4
TRAIN_MANIFEST="train_manifest.csv"
EVAL_MANIFEST="eval_manifest.csv"
DEVICE="cuda"
MAX_FEATURES=""
# LR schedule
SCHEDULER="cosine_warmup"       # constant | cosine_warmup | linear_warmup | cosine_with_restarts | polynomial_decay_warmup
WARMUP_PROPORTION="0.1"         # fraction of max_steps used for linear warmup (0.0 = no warmup)

# ---- Wandb ------------------------------------------------------------------
WANDB_LOG=1
WANDB_PROJECT="TabICL"
WANDB_NAME="GP_stage3_hpo"
WANDB_MODE="offline"

# ---- Reproducibility --------------------------------------------------------
NP_SEED=42
TORCH_SEED=42

# ---- Parallelism ------------------------------------------------------------
#
# MAX_CONCURRENT – hard cap on the total number of HPO worker processes that
#   may be alive at the same time.
#
#   0  (default) = auto: the cap is set to  detected_gpus × RUNS_PER_GPU
#                  when DEVICE starts with "cuda", or 1 when using CPU.
#                  Use this in almost all cases – it saturates every GPU slot
#                  without over-subscribing.
#
#   N > 0        = explicit override: always allow exactly N concurrent runs,
#                  regardless of how many GPUs are detected or what RUNS_PER_GPU
#                  is set to.  Useful if you want to deliberately under-subscribe
#                  (e.g. MAX_CONCURRENT=1 to run purely sequentially for
#                  debugging) or if you know the auto-detection will be wrong.
#
MAX_CONCURRENT=0
#
# RUNS_PER_GPU – how many worker processes may share a single GPU simultaneously.
#   Only meaningful when MAX_CONCURRENT=0 (auto mode) and DEVICE=cuda.
#
#   1  (conservative) = one run per GPU at a time; safe for any model size.
#   2+                = pack multiple small models onto one GPU to keep it busy.
#                       With a 26M-parameter model on a 32 GB V100 you can
#                       comfortably use 2–4 here.  Increase until you see OOM
#                       errors or GPU memory > ~90 % in nvidia-smi, then back off.
#
#   With one GPU and RUNS_PER_GPU=2:  MAX_CONCURRENT resolves to 2.
#   With two GPUs and RUNS_PER_GPU=3: MAX_CONCURRENT resolves to 6.
#
RUNS_PER_GPU=1

# =============================================================================
# Parse optional command-line overrides
# =============================================================================

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --data-dir            PATH
  --checkpoint-path     PATH
  --checkpoint-dir      PATH
  --wandb-dir           PATH
  --lr-values           CSV_FLOATS
  --max-steps-values    CSV_INTS
  --batch-size-values   CSV_INTS
  --weight-decay-values CSV_FLOATS
  --proximal-wd-values  CSV  comma-separated true/false values (e.g. false,true to sweep both)
  --micro-batch-size    INT
  --save-perm-every     INT
  --save-temp-every     INT
  --n-eval              INT
  --eval-seed           INT
  --train-manifest      STR
  --eval-manifest       STR
  --device              STR
  --max-features                INT
  --scheduler                   STR   constant|cosine_warmup|linear_warmup|cosine_with_restarts|polynomial_decay_warmup
  --warmup-proportion           FLOAT fraction of max_steps for LR warmup (0.0 = none, -1 = use fixed warmup_steps)
  --num-additional-checkpoints  INT   extra periodic ckpts evaluated at Stage-2 (0=best only, median over N+1)
  --wandb-log                         enable wandb logging
  --wandb-project       STR
  --wandb-name          STR
  --wandb-mode          STR   online|offline|disabled
  --np-seed             INT
  --torch-seed          INT
  --max-concurrent      INT   0=auto
  --runs-per-gpu        INT   simultaneous runs per detected GPU
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)            DATA_DIR="$2";            shift 2 ;;
    --checkpoint-path)     INIT_CHECKPOINT="$2";     shift 2 ;;
    --checkpoint-dir)      CHECKPOINT_DIR="$2";      shift 2 ;;
    --wandb-dir)           WANDB_DIR="$2";           shift 2 ;;
    --lr-values)           LR_VALUES="$2";           shift 2 ;;
    --max-steps-values)    MAX_STEPS_VALUES="$2";    shift 2 ;;
    --batch-size-values)   BATCH_SIZE_VALUES="$2";   shift 2 ;;
    --weight-decay-values)  WEIGHT_DECAY_VALUES="$2";  shift 2 ;;
    --proximal-wd-values)   PROXIMAL_WD_VALUES="$2";   shift 2 ;;
    --micro-batch-size)     MICRO_BATCH_SIZE="$2";     shift 2 ;;
    --save-perm-every)     SAVE_PERM_EVERY="$2";     shift 2 ;;
    --save-temp-every)     SAVE_TEMP_EVERY="$2";     shift 2 ;;
    --n-eval)              N_EVAL="$2";              shift 2 ;;
    --eval-seed)           EVAL_SEED="$2";           shift 2 ;;
    --train-manifest)      TRAIN_MANIFEST="$2";      shift 2 ;;
    --eval-manifest)       EVAL_MANIFEST="$2";       shift 2 ;;
    --device)              DEVICE="$2";              shift 2 ;;
    --max-features)        MAX_FEATURES="$2";        shift 2 ;;
    --scheduler)           SCHEDULER="$2";           shift 2 ;;
    --warmup-proportion)   WARMUP_PROPORTION="$2";   shift 2 ;;
    --wandb-log)           WANDB_LOG=1;               shift 1 ;;
    --wandb-project)       WANDB_PROJECT="$2";       shift 2 ;;
    --wandb-name)          WANDB_NAME="$2";          shift 2 ;;
    --wandb-mode)          WANDB_MODE="$2";          shift 2 ;;
    --np-seed)             NP_SEED="$2";             shift 2 ;;
    --torch-seed)          TORCH_SEED="$2";          shift 2 ;;
    --max-concurrent)      MAX_CONCURRENT="$2";      shift 2 ;;
    --runs-per-gpu)                 RUNS_PER_GPU="$2";                 shift 2 ;;
    --num-additional-checkpoints)   NUM_ADDITIONAL_CHECKPOINTS="$2";   shift 2 ;;
    -h|--help)             usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "ERROR: --data-dir is required (or set DATA_DIR in script)."
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR" "$WANDB_DIR"

CMD=(
  python "$PY_SCRIPT"
  --data_dir "$DATA_DIR"
  --checkpoint_dir "$CHECKPOINT_DIR"
  --wandb_dir "$WANDB_DIR"
  --lr_values "$LR_VALUES"
  --max_steps_values "$MAX_STEPS_VALUES"
  --batch_size_values "$BATCH_SIZE_VALUES"
  --weight_decay_values "$WEIGHT_DECAY_VALUES"
  --proximal_wd_values "$PROXIMAL_WD_VALUES"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --save_perm_every "$SAVE_PERM_EVERY"
  --save_temp_every "$SAVE_TEMP_EVERY"
  --n_eval "$N_EVAL"
  --eval_seed "$EVAL_SEED"
  --train_manifest "$TRAIN_MANIFEST"
  --eval_manifest "$EVAL_MANIFEST"
  --device "$DEVICE"
  --scheduler "$SCHEDULER"
  --warmup_proportion "$WARMUP_PROPORTION"
  --wandb_project "$WANDB_PROJECT"
  --wandb_name "$WANDB_NAME"
  --wandb_mode "$WANDB_MODE"
  --np_seed "$NP_SEED"
  --torch_seed "$TORCH_SEED"
  --max_concurrent "$MAX_CONCURRENT"
  --runs_per_gpu "$RUNS_PER_GPU"
  --num_additional_checkpoints "$NUM_ADDITIONAL_CHECKPOINTS"
)

if [[ -n "$INIT_CHECKPOINT" ]]; then
  CMD+=(--checkpoint_path "$INIT_CHECKPOINT")
fi

if [[ "$WANDB_LOG" -eq 1 ]]; then
  CMD+=(--wandb_log)
fi

if [[ -n "$MAX_FEATURES" ]]; then
  CMD+=(--max_features "$MAX_FEATURES")
fi

echo "============================================================"
echo "  TabICL GP Stage-3 HPO"
echo "  DATA_DIR            : $DATA_DIR"
echo "  CHECKPOINT_DIR      : $CHECKPOINT_DIR"
echo "  WANDB_DIR           : $WANDB_DIR"
echo "  LR_VALUES           : $LR_VALUES"
echo "  MAX_STEPS_VALUES    : $MAX_STEPS_VALUES"
echo "  BATCH_SIZE_VALUES   : $BATCH_SIZE_VALUES"
echo "  WEIGHT_DECAY_VALUES : $WEIGHT_DECAY_VALUES"
echo "  PROXIMAL_WD_VALUES  : $PROXIMAL_WD_VALUES"
echo "  DEVICE              : $DEVICE"
echo "  SCHEDULER           : $SCHEDULER"
echo "  WARMUP_PROPORTION   : $WARMUP_PROPORTION"
echo "  RUNS_PER_GPU        : $RUNS_PER_GPU"
echo "  MAX_CONCURRENT      : $MAX_CONCURRENT"
echo "  NUM_ADDITIONAL_CKPT : $NUM_ADDITIONAL_CHECKPOINTS"
echo "============================================================"

"${CMD[@]}"
