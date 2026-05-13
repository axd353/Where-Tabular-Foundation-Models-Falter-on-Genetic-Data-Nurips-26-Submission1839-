# Data Generation

This folder is a minimal copied subset of the data-generation source tree for producing the mixed HGP + PRS-CSx distribution-shift data used by the TabICL instruction-tuning experiments.

## Copied Components

- `test_train_dist_shift_mixed.py`: mixed-family orchestrator.
- `test_train_dist_shift.py`: HGP generator.
- `make_data_ver2.py`, `make_data_ver3.py`: simulator utilities used by HGP and PRS-CSx stages.
- `PRS-CSx/`: PRS-CSx simulator and stage modules used by the mixed orchestrator.

## Reference Run

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

Resume an interrupted run with the same command plus `--resume`.

## Customising task settings

Per-task simulation hyperparameters (population size `M`, number of clusters, ancestry-group structure, GP lengthscales, MAF ranges, PRS-CSx specific `phi`/`r_g`/`pi_causal`/`alpha`/`h`, etc.) are sampled inside `sample_setting()`. Edit:

- HGP tasks: [test_train_dist_shift.py](test_train_dist_shift.py) → `sample_setting()`
- PRS-CSx tasks: [PRS-CSx/test_train_dist_shift_prs_csx.py](PRS-CSx/test_train_dist_shift_prs_csx.py) → `sample_setting()`

The mixed orchestrator imports both files, so editing either `sample_setting()` is sufficient to change the corresponding family in the mixed run.
