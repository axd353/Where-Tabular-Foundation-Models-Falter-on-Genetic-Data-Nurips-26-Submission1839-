# NeurIPS Submission Code

This standalone repository contains the code needed for the paper's GP/PRS-CSx data generation and TabICL instruction-tuning flow.

The implementation files are copied from the working source trees without editing the originals:

- `DataGeneration/`: mixed HGP + PRS-CSx dataset generation.
- `InstructionTuneTabicl/`: TabICL GP Stage-3 HPO / instruction-tuning flow.

## Environment

The reference environment lives at `/nfs/home/adas23/python_environments/tabpfn_latest/` (Python 3.9.10, PyTorch 2.8.0 + CUDA 12.8, TabPFN 6.3.2). Full step-by-step setup, including the exact `torch` install command and pinned package versions, is documented at the top of [requirements.txt](requirements.txt). The short version is:

```bash
python3.9 -m venv /nfs/home/adas23/python_environments/tabpfn_latest
source /nfs/home/adas23/python_environments/tabpfn_latest/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch first using the CUDA 12.8 wheels (or the CPU index).
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

cd code_for_paper
python -m pip install -r requirements.txt
```

If the environment already exists, just `source .../bin/activate` and rerun the final install command from this repo root. The `tabicl` package is imported directly from `InstructionTuneTabicl/src` via `sys.path` injection inside the scripts, so it does not need to be pip-installed.

## Data Generation

```bash
source /nfs/home/adas23/python_environments/tabpfn_latest/bin/activate
cd code_for_paper/DataGeneration

DATA_DIR="<path to where data gets stored>"

python test_train_dist_shift_mixed.py \
  "$DATA_DIR" \
  --num_train_settings 400 \
  --num_eval_settings 90 \
  --base_seed 1990 \
  --n_proc 39
```

Use `--resume` to continue an interrupted generation run. The output directory will contain `train_manifest.csv`, `eval_manifest.csv`, generated pickles, and `generation.log`.

## Instruction Tuning / HPO

The HPO shell script keeps its original defaults, so pass explicit output directories when running from this clean repo:

```bash
source /nfs/home/adas23/python_environments/tabpfn_latest/bin/activate
cd code_for_paper/InstructionTuneTabicl/test_train_dist_shift_exps

DATA_DIR="<path to where data gets stored>"

bash run_gp_hpo.sh \
  --data-dir "$DATA_DIR" \
  --checkpoint-dir ../checkpoints \
  --wandb-dir ../wandb_logs \
  --lr-values "1e-7,5e-7,9e-7,1e-6,2e-6,2.5e-6,4.5e-6,6e-6,8e-6,1e-5" \
  --max-steps-values "500" \
  --batch-size-values "10" \
  --weight-decay-values "1e-2,1e-1" \
  --proximal-wd-values "true" \
  --micro-batch-size 1 \
  --n-eval 30 \
  --num-additional-checkpoints 4 \
  --runs-per-gpu 1 \
  --wandb-mode offline
```

If the TabICL checkpoint is not already cached from Hugging Face, add `--checkpoint-path /path/to/tabicl-classifier.ckpt`.

See `InstructionTuneTabicl/test_train_dist_shift_exps/HPO_FLOW_GUIDE.md` for the detailed training and evaluation flow.

## Downloadable archive

A zipped snapshot of this repository is included as `paper1839-code.zip` for reviewers who prefer a single download instead of browsing files. It is generated from the latest commit via `git archive --format=zip --output=paper1839-code.zip HEAD`.
