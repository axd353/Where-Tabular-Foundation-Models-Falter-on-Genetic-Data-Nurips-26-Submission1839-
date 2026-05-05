# GP Stage-3 HPO Flow Guide

**Initiated by:** `test_train_dist_shift_exps/run_gp_hpo.sh`  
**Reference run:** `logs/hpo_runs/20260306_005601` / `models/hpo_runs/20260306_005601`

---

## Broad Overview

The shell script launches a **hyperparameter optimisation (HPO) sweep** that
fine-tunes the TabICL model on a collection of GP-simulation genetics datasets.
The goal is to adapt (Stage-3 fine-tune) TabICL's in-context-learning head
(`tf_icl`) to handle the distribution shift between training and test ancestry
groups that occurs in real genetic risk-score tasks.

**Stage-3 freezing (from the TabICL paper):** TabICL is composed of three modules:
a column embedder (`col_embedder`), a row interactor (`row_interactor`), and
an ICL-learning head (`tf_icl`). In Stage-3 fine-tuning — as described in
*TabICL: A Tabular Foundation Model for In-Context Learning on Large Data* —
only the `tf_icl` head is updated; the column embedder and row interactor are
**frozen** (`arch_freeze_col=True`, `arch_freeze_row=True`). This preserves
the general-purpose feature and row representations learned in earlier training
stages while allowing the in-context learning head to specialise toward the
distribution-shifted genetics setting.

The overall flow has three phases:

```
run_gp_hpo.sh
  └─► train_gp_hpo.py  [Orchestrator – _main_hpo()]
        • Builds Cartesian-product grid  (LR × max_steps × batch_size × weight_decay × proximal_wd)
        • Creates a dated run directory  (e.g. 20260306_005601/)  under both CHECKPOINT_DIR and WANDB_DIR
        • Writes hpo_runs_manifest.csv and hpo_fixed_config.csv
        • Spawns one child process per grid cell:

            train_gp_hpo.py --single_run ...  [Worker – _single_run()]
              └─► _build_tabicl_config()          (from train_gp.py)
              └─► GPTrainer.train()               (from gp_trainer.py)
                    • Loads a pre-trained TabICL checkpoint
                    • Replaces the data prior with GPPriorDataset  (gp_dataset.py)
                    • Runs max_steps gradient updates
                    • After every step: evaluates on a fixed Stage-1 eval batch
                    • Saves best-eval checkpoint whenever eval accuracy improves

        • After ALL workers finish:
            hpo_stage2_eval.run_stage2_eval()    (from hpo_stage2_eval.py)
              • Evaluates the best checkpoint of every run on a held-out Stage-2 slice
              • Appends stage2_eval_acc / stage2_eval_ce to hpo_runs_manifest.csv
```

Workers **share GPUs** via the `CUDA_VISIBLE_DEVICES` environment variable.
With `RUNS_PER_GPU=9` the orchestrator can assign up to 9 processes to a
single GPU simultaneously (useful for small models; tune down if OOM).

---

## 1. Train and Eval Manifests — What They Are and How They're Used

### What the manifests contain

Each manifest is a CSV file inside `DATA_DIR`.  Every row describes one
simulated genetics dataset produced by the GP (Gaussian-Process) prior:

| Column | Meaning |
|---|---|
| `path_to_train` | Absolute path to a **train pickle** (≥5000 individuals) |
| `path_to_test` | Absolute path to a **test pickle** (≥100 individuals) |
| `A` | Number of ancestry/PCA features |
| `N` | Number of SNP/genotype features |
| `M`, `num_clusters`, `seed`, … | Simulation hyper-parameters (bookkeeping only) |

The **train** pickle (`distshift_train_NNNN_seed*_train.pkl`) contains
individuals drawn from the **training** ancestry distribution;  
the **test** pickle (`distshift_train_NNNN_seed*_test.pkl`) contains
individuals drawn from the **test** ancestry distribution — i.e., a
different population than what the labels were learned from (the
distribution-shift setting).

Both pickles contain a pandas DataFrame under key `"table"` with columns:
`y` (binary label), `cluster_id`, `sample_index`, and `A + N` feature columns.

In the reference run: `train_manifest.csv` has **300 rows**, `eval_manifest.csv`
has **100 rows**.

### How manifest rows become training tasks

`GPPriorDataset` (`gp_dataset.py`) is the PyTorch `IterableDataset` used
during training. Each call to `get_batch(batch_size)` **randomly samples
`batch_size` rows** from `train_manifest.csv` (with or without replacement)
and converts each row into one tabular in-context learning (ICL) task via
`_build_task()`:

```
For each sampled manifest row:
  1. Load train pickle  → DataFrame with N_train_total individuals
  2. Load test  pickle  → DataFrame with N_test_total  individuals
  3. Sample  N_tst = 0.9 × N_test_total  test rows (without replacement)
  4. Sample  N_train_needed = 10 × N_tst  train rows (with replacement if needed)
  5. Concatenate: [ train_context (N_train_needed rows) | test_query (N_tst rows) ]
  6. Pad feature dimension to max_features with zeros
  7. Remap y labels to contiguous 0-indexed integers
```

**In-context examples (context tokens):** the `N_train_needed` train-pickle
rows placed at the **front** of the sequence.  Their features **and** their
labels `y` are both visible to the model.

**Unlabelled query (what gets predicted):** the `N_tst` test-pickle rows at
the **back** of the sequence.  Features are provided but labels are masked
(the forward pass predicts them; loss is computed only on these positions).

The tabicl model sees the full concatenated sequence `[context | query]` in
one forward pass, reads the context labels autoregressively, and predicts the
query labels.

### MAX_STEPS_VALUES — steps and gradient updates

`MAX_STEPS_VALUES="400"` sets `max_steps=400` for every run in this
sweep. Each **step** is one gradient update:

```
for step in range(max_steps):          # = 400 iterations
    batch = next(dataloader)           # batch_size tasks sampled from train_manifest
    run_batch(batch)                   # gradient update
    _eval_log_and_checkpoint(step)     # Stage-1 eval on fixed eval batch
```

Each gradient update processes `batch_size` tasks split into micro-batches
of `micro_batch_size` tasks each:

```
num_micro_batches = batch_size // micro_batch_size  = 10 // 1 = 10
for micro_batch in split(batch, micro_batch_size):
    forward → compute CE loss / 10   (scaled by 1/num_micro_batches)
    backward (accumulate gradients)
optimizer.step()                      # ONE weight update after all 10 micro-batches
```

**Total gradient updates per run: 400** (one per step, regardless of
`batch_size` or `micro_batch_size`).

### BATCH_SIZE_VALUES and MICRO_BATCH_SIZE

| Parameter | Value | Meaning |
|---|---|---|
| `BATCH_SIZE_VALUES` | `10` | Number of manifest tasks (ICL problems) per gradient update |
| `MICRO_BATCH_SIZE` | `1` | Number of tasks per forward/backward pass (gradient accumulation) |

`BATCH_SIZE_VALUES` controls the **effective batch size** — how diverse the
signal for each gradient step is and how large the gradient estimate is.
With `batch_size=10` the model sees 10 different GP datasets per update.

`MICRO_BATCH_SIZE=1` means gradient accumulation: the 10-task batch is
processed one task at a time, gradients accumulate, and the optimizer steps
once at the end.  This reduces peak GPU memory without changing the
mathematical gradient (relative to processing all 10 simultaneously).

Each task can produce variable-length sequences (different manifests have
different `A+N` and different sampled sizes).  Using `micro_batch_size=1`
also sidesteps the variable-length padding problem across tasks within a batch.

> **Concrete numbers for the reference run:**  
> Each of the 400 steps processes 10 train-manifest tasks (each with
> ~90 test-query rows and ~900 context rows), one at a time, accumulating
> gradients before a single Adam update.

---

## 2. Stage-1 and Stage-2 Evaluation Accuracy

### Stage-1 Evaluation (during training)

**When:** evaluated at every gradient-update step, *before* the step's weight
update (so step 0 = pre-training baseline).

**Data:** first `n_eval` rows of `eval_manifest.csv`.  In the reference run
`n_eval=30`, using eval-manifest rows **0 … 29**.

The eval batch is **fixed and pre-generated once** at the start of training
(`GPTrainer._prepare_eval_data()`).  The same 30 tasks are reused at every
step so that metrics are directly comparable across steps.  The eval tasks
are constructed with the same `_build_task()` logic as training tasks (same
90/10 test-subsample, same 10× train scale), but with a **separate fixed seed**
(`eval_seed=0`) so the samples are reproducible and identical each time.

**In-context examples:** rows from the **train pickle** of each eval-manifest
row (same ancestry-shift structure as training tasks).  
**Query / unlabelled examples:** rows from the **test pickle** of each
eval-manifest row.

**Metric computed — plain accuracy (not ROC-AUC):**

```python
# _eval_single_task() in gp_trainer.py
pred = model(X, y_train, d)           # (1, test_size, n_classes)
pred = pred.flatten(end_dim=-2)       # (test_size, n_classes)
true = y_test.long().flatten()        # (test_size,)
loss = F.cross_entropy(pred, true)    # scalar CE
acc  = (pred.argmax(dim=1) == true).float().mean()  # hard-label accuracy
```

For each step the **mean CE** and **mean hard-label accuracy** across all 30
eval tasks is computed (macro-average over tasks, micro-average within each
task).

**Where it is used:**
- Logged to Weights & Biases as `eval_ce` and `eval_accuracy` every step.
- Drives best-checkpoint selection: whenever `eval_accuracy` exceeds the
  previous best, `best-eval-step-<N>.ckpt` is saved (previous best deleted).

### Stage-2 Evaluation (post-training, once per run)

**When:** after **all** HPO worker processes have finished,
the orchestrator calls `run_stage2_eval()` from `hpo_stage2_eval.py`.

**Data:** a **held-out slice** of `eval_manifest.csv` that was never seen
during training.  Specifically:

```
total eval rows       = 100
Stage-1 rows used     = n_eval = 30     (rows 0 … 29)
remaining             = 70
n_stage2              = 70 // 2 = 35    (first half of remainder)
Stage-2 rows          = rows 30 … 64   (last_row_used = 64)
rows 65 … 99 reserved for future Stage-3 evaluation
```

This matches `stage2_last_row_used=64` seen in the reference manifest.

**Checkpoint(s) evaluated:** with `NUM_ADDITIONAL_CHECKPOINTS=4`, for each run:
1. The saved `best-eval-step-<N>.ckpt` (best Stage-1 checkpoint).
2. The 4 periodic `step-*.ckpt` files whose step number is closest to the
   best checkpoint's step (selected by `_find_evaluation_checkpoints()`).

The model is loaded fresh for each checkpoint via `_EvalOnlyTrainer`.

**Metric computed — same plain accuracy + CE:**

Each of the `(1 + 4) = 5` checkpoints is run through `_run_eval()` on the 35
Stage-2 tasks.  The **median** over the 5 per-checkpoint accuracies and the
median over the 5 CEs are stored:

```
stage2_eval_acc = median( [acc_best, acc_periodic_1, …, acc_periodic_4] )
stage2_eval_ce  = median( [ce_best,  ce_periodic_1,  …, ce_periodic_4] )
```

Taking the median over nearby checkpoints reduces noise from step-to-step
training variance in the HPO ranking without biasing toward any single
checkpoint's "lucky" step.

**Where it is used:**
- Appended to `hpo_runs_manifest.csv` as three new columns:
  `stage2_eval_acc`, `stage2_eval_ce`, `stage2_last_row_used`.
- Intended as the **primary HPO ranking criterion** — it is computed on data
  the model never saw, so it is a less biased estimate of generalisation.
- The `visualize_hpo_training.ipynb` notebook reads these columns to rank and
  plot runs.

> **Neither stage uses ROC-AUC.**  Both use hard-label classification
> accuracy (`argmax` match rate) and cross-entropy loss.  Because all datasets
> are binary classification (y ∈ {0,1}), accuracy is a reasonable proxy, but
> note it does not account for predicted probability calibration.

---

## 3. What Is Stored in the Log CSVs

### `hpo_runs_manifest.csv`

**Location:** written into both `WANDB_DIR/<run_id>/` and `CHECKPOINT_DIR/<run_id>/`.  
**One row per HPO run**, created at orchestrator startup and later updated
with Stage-2 results.

Columns written **at startup** (by `_write_manifest()`):

| Column | Example | Description |
|---|---|---|
| `run_name` | `run_001__lr_1e-07__max_steps_400__batch_size_10__weight_decay_0__proximal_wd_False` | Unique identifier; encodes all swept HPs |
| `lr` | `1e-07` | Learning rate for this run |
| `max_steps` | `400` | Number of gradient update steps |
| `batch_size` | `10` | Tasks per gradient update |
| `weight_decay` | `0.0` | AdamW L2 penalty (or proximal λ) |
| `proximal_wd` | `False` | Whether proximal regularisation is active |
| `checkpoint_dir` | `/…/models/hpo_runs/20260306_005601/run_001__…/` | Per-run checkpoint directory |
| `wandb_dir` | `/…/logs/hpo_runs/20260306_005601/run_001__…/` | Per-run W&B / log directory |

Columns **appended after Stage-2** (by `run_stage2_eval()`):

| Column | Example | Description |
|---|---|---|
| `stage2_eval_acc` | `0.7442` | Median accuracy over Stage-2 tasks (across num_additional+1 checkpoints) |
| `stage2_eval_ce` | `0.4920` | Median CE loss over Stage-2 tasks |
| `stage2_last_row_used` | `64` | Last eval-manifest row index consumed by Stage-1 + Stage-2 combined |

### `hpo_fixed_config.csv`

**Location:** same two directories as the manifest.  
**Three columns:** `config_name`, `value`, `description`.  
Records everything that was **constant across all runs** in this sweep:

- Fixed training options: `data_dir`, `init_checkpoint_path`, `micro_batch_size`,
  `save_perm_every`, `save_temp_every`, `n_eval`, `eval_seed`, `train_manifest`,
  `eval_manifest`, `device`, `max_features`, `scheduler`, `warmup_proportion`,
  `wandb_*`, seeds, parallelism settings.
- Architecture constants: `embed_dim=128`, `col_num_blocks=3`, `row_num_blocks=3`,
  `icl_num_blocks=12`, `freeze_col=True`, `freeze_row=True` (Stage-3 only trains `tf_icl`), etc.

This makes the run directory **self-contained** — you can reconstruct the exact
experimental conditions without reading the shell script.

---

## 4. How Metric Quantities Are Computed — Step by Step

```
Training (run_micro_batch in gp_trainer.py):
──────────────────────────────────────────────
  micro_X,  micro_y,  micro_d,  micro_seq_len,  micro_train_size = micro_batch
  y_train = micro_y[:, :train_size]     # context labels  (visible)
  y_test  = micro_y[:, train_size:]     # query labels    (predicted)
  pred    = model(micro_X, y_train, micro_d)         # (B, test_size, n_classes)
  ce_loss = F.cross_entropy(pred.flatten(-2), y_test.long().flatten())
  train_accuracy = (pred.argmax(-1) == y_test.long()).float().mean()

  # With proximal_wd=True, an extra term is added:
  proximal_loss = (weight_decay / 2) * Σ_i ||θ_i − θ_anchor_i||²
  loss_total    = ce_loss + proximal_loss
  (ce_loss alone is logged; proximal_loss is logged separately)

Evaluation (_eval_single_task in gp_trainer.py):
──────────────────────────────────────────────────
  # Called with no_grad, model in eval mode
  pred  = model(X, y_train, d)          # (1, test_size, n_classes)
  loss  = F.cross_entropy(pred.flatten(-2), true)   # CE over query positions
  acc   = (pred.argmax(-1) == true).float().mean()  # fraction of correct predictions

  # Averaged over n_eval tasks (30 for Stage-1, 35 for Stage-2):
  eval_ce       = mean(loss per task)
  eval_accuracy = mean(acc  per task)   ← plain hard-label accuracy, NOT ROC-AUC
```

---

## 5. Relevant Code Files and Their Roles

| File | Role | Key functions/classes |
|---|---|---|
| `run_gp_hpo.sh` | Entry point; sets all config variables and invokes `train_gp_hpo.py` | — |
| `train_gp_hpo.py` | Orchestrator + worker dispatcher | `_main_hpo()`, `_single_run()`, `_launch_run()`, `_write_manifest()`, `_write_fixed_config_csv()` |
| `train_gp.py` | Config builder; standalone single-run training script | `_build_tabicl_config()`, `_resolve_checkpoint()` |
| `gp_trainer.py` | Training loop (GPTrainer subclasses tabicl's Trainer) | `GPTrainer.train()`, `configure_prior()`, `_prepare_eval_data()`, `_run_eval()`, `_eval_log_and_checkpoint()`, `run_micro_batch()` |
| `gp_dataset.py` | Dataset that serves ICL tasks from disk pickles | `GPPriorDataset`, `_build_task()`, `prepare_eval_batch()` |
| `hpo_stage2_eval.py` | Post-hoc held-out evaluation; updates manifest CSV | `run_stage2_eval()`, `prepare_stage2_eval_batch()`, `_find_evaluation_checkpoints()`, `_EvalOnlyTrainer` |
| `visualize_hpo_training.ipynb` | Reads manifest CSVs and plots HPO results | — |

### Call chain during a complete sweep

```
run_gp_hpo.sh
  └─► python train_gp_hpo.py  (orchestrator)
        ├── _main_hpo(args)
        │     ├── _write_manifest()           → hpo_runs_manifest.csv
        │     ├── _write_fixed_config_csv()   → hpo_fixed_config.csv  [written AFTER all runs finish]
        │     ├── for each grid cell:
        │     │     └── _launch_run()
        │     │           └── subprocess: python train_gp_hpo.py --single_run ...
        │     │                 └── _single_run(args)
        │     │                       ├── _build_tabicl_config()    [train_gp.py]
        │     │                       └── GPTrainer(config).train() [gp_trainer.py]
        │     │                             ├── configure_optimizer()
        │     │                             ├── load_checkpoint()       ← TabICL weights
        │     │                             ├── configure_prior()       ← GPPriorDataset [gp_dataset.py]
        │     │                             ├── _prepare_eval_data()    ← fixed Stage-1 eval batch [gp_dataset.py]
        │     │                             └── for step in range(max_steps):
        │     │                                   _eval_log_and_checkpoint(step)  ← Stage-1 eval
        │     │                                   batch = next(dataloader)        ← train task batch
        │     │                                   run_batch(batch)                ← gradient update
        │     │                                   save_checkpoint() if save_*
        │     │                             └── _eval_log_and_checkpoint(max_steps)  ← post-train eval
        │     │
        │     └── run_stage2_eval()           [hpo_stage2_eval.py]
        │           ├── prepare_stage2_eval_batch()   ← held-out eval rows [gp_dataset.py via _build_task]
        │           ├── for each run in manifest:
        │           │     ├── _find_evaluation_checkpoints()  ← best + 4 nearby periodic ckpts
        │           │     └── _EvalOnlyTrainer.evaluate_tasks()
        │           │           └── GPTrainer._run_eval()     ← _eval_single_task() per task
        │           └── manifest.to_csv()    ← stage2_* columns appended
        │
        └── _write_fixed_config_csv()         → hpo_fixed_config.csv
```

