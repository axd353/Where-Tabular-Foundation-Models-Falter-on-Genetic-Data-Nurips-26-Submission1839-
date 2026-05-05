# InstructionTuneTabicl

This folder is a copied subset of the TabICL source tree for the GP Stage-3 TabICL instruction-tuning / HPO flow.

The original experiment directory depth is preserved because `train_gp.py` and `hpo_stage2_eval.py` resolve `src/tabicl` relative to `test_train_dist_shift_exps/`.

## Copied Components

- `test_train_dist_shift_exps/run_gp_hpo.sh`: HPO entry point.
- `test_train_dist_shift_exps/train_gp_hpo.py`: HPO orchestrator and worker launcher.
- `test_train_dist_shift_exps/train_gp.py`: TabICL config builder and checkpoint resolution.
- `test_train_dist_shift_exps/gp_trainer.py`: GP Stage-3 trainer.
- `test_train_dist_shift_exps/gp_dataset.py`: manifest-backed GP prior dataset.
- `test_train_dist_shift_exps/hpo_stage2_eval.py`: held-out Stage-2 HPO evaluation.
- `test_train_dist_shift_exps/HPO_FLOW_GUIDE.md`: detailed flow guide copied from the source repo.
- `src/tabicl/`: TabICL model and training package.

## Run

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

If the public TabICL checkpoint is not already cached, add `--checkpoint-path /path/to/tabicl-classifier.ckpt`.
