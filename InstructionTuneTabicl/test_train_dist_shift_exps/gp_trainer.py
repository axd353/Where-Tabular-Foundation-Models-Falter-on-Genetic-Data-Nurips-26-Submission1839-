"""
gp_trainer.py
=============
GPTrainer: a Trainer subclass that

  1. Replaces the standard PriorDataset with GPPriorDataset loaded from disk.
  2. Pre-generates a fixed evaluation batch (first N rows of eval_manifest.csv)
     before the training loop.
  3. After every gradient-update step, evaluates the current model on that
     fixed batch (no_grad, eval mode) and logs results to wandb.
  4. Saves a "best" checkpoint whenever the evaluation accuracy improves.

The extra configuration attributes read from config (with defaults):

  config.gp_data_dir          – path to root data directory  (required)
  config.gp_train_manifest    – filename, default "train_manifest.csv"
  config.gp_eval_manifest     – filename, default "eval_manifest.csv"
  config.gp_n_eval            – number of eval datasets,  default 10
  config.gp_eval_seed         – seed for eval sampling,   default 0
  config.gp_max_features      – int or None (auto-computed if None)
  config.gp_cache_pickles     – bool, default True
  config.gp_proximal_wd       – bool, default False
                                 When True, weight decay regularises toward the
                                 initial checkpoint weights instead of toward zero
                                 (proximal / anchor regularisation).  AdamW's
                                 built-in L2 penalty is disabled and replaced by
                                 the manual term:  (weight_decay/2) * ||θ−θ₀||²
                                 appended to the cross-entropy loss each step.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# TabICL imports (resolved via sys.path set by the launch script)
from tabicl.train.run import Trainer, ddp_cleanup, Timer

from gp_dataset import GPPriorDataset, prepare_eval_batch


# ---------------------------------------------------------------------------
# Helper: evaluate a single pre-generated task
# ---------------------------------------------------------------------------

def _eval_single_task(
    model: torch.nn.Module,
    X_i: Tensor,
    y_i: Tensor,
    n_feat: int,
    train_size: int,
    device: str,
) -> Tuple[float, float]:
    """
    Run one eval task through the model (no_grad).

    Parameters
    ----------
    model      : TabICL model (already in eval mode, no DDP wrapper needed)
    X_i        : (T, max_features) float32
    y_i        : (T,)              float32
    n_feat     : actual number of features for this dataset
    train_size : split point between context and query
    device     : target device string

    Returns
    -------
    (ce_loss, accuracy)  – scalars, Python floats
    """
    # Add batch dimension  →  (1, T, max_features) / (1, T)
    X  = X_i.unsqueeze(0).to(device)        # (1, T, max_features)
    y  = y_i.unsqueeze(0).to(device)        # (1, T)   float32
    d  = torch.tensor([n_feat], dtype=torch.long, device=device)  # (1,)

    # Truncate to actual sequence length and feature count
    seq_len = X.shape[1]
    X = X[:, :seq_len, :n_feat]

    y_train = y[:, :train_size]             # (1, train_size)
    y_test  = y[:, train_size:]             # (1, test_size)

    with torch.no_grad():
        pred = model(X, y_train, d)         # (1, test_size, max_classes)
        pred = pred.flatten(end_dim=-2)     # (test_size, max_classes)
        true = y_test.long().flatten()      # (test_size,)
        loss = F.cross_entropy(pred, true)
        acc  = (pred.argmax(dim=1) == true).float().mean()

    return float(loss.item()), float(acc.item())


# ---------------------------------------------------------------------------
# GPTrainer
# ---------------------------------------------------------------------------

class GPTrainer(Trainer):
    """
    Trainer subclass for fine-tuning TabICL on GP-simulation datasets.

    Only configure_prior and train are overridden; everything else
    (model build, optimiser, checkpoint loading, wandb setup) is inherited
    from the base Trainer.
    """

    # ------------------------------------------------------------------
    def configure_optimizer(self):
        """
        Configure optimizer and scheduler.

        When ``config.gp_proximal_wd`` is True the AdamW weight-decay
        parameter is set to 0 so that the optimiser does *not* shrink weights
        toward zero; the proximal regularisation term added in
        ``run_micro_batch`` handles the actual regularisation instead.
        The ``config.weight_decay`` value is preserved for use as 'λ' in
        that term.
        """
        use_proximal = getattr(self.config, "gp_proximal_wd", False)
        if use_proximal:
            # Temporarily zero out weight_decay so AdamW does not apply
            # decay-to-zero; proximal regularization is applied manually.
            original_wd = self.config.weight_decay
            self.config.weight_decay = 0.0
            super().configure_optimizer()
            self.config.weight_decay = original_wd  # restore for proximal use
            if self.master_process:
                print(
                    f"[ProximalWD] AdamW weight_decay set to 0; "
                    f"proximal λ={original_wd} will be applied manually."
                )
        else:
            super().configure_optimizer()

    # ------------------------------------------------------------------
    def load_checkpoint(self):
        """
        Load checkpoint via the base class, then – if proximal weight decay
        is enabled – freeze a copy of every trainable parameter as the
        anchor θ₀ toward which the proximal term will regularise.
        """
        super().load_checkpoint()
        if getattr(self.config, "gp_proximal_wd", False):
            self._anchor_params: Optional[List[torch.Tensor]] = [
                p.data.clone().detach()
                for p in self.raw_model.parameters()
                if p.requires_grad
            ]
            if self.master_process:
                n = sum(a.numel() for a in self._anchor_params)
                print(
                    f"[ProximalWD] Anchor captured from checkpoint: "
                    f"{n:,} trainable parameter elements."
                )
        else:
            self._anchor_params = None

    # ------------------------------------------------------------------
    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        """
        Process one micro-batch, optionally adding a proximal regularisation
        term to the loss.

        When ``config.gp_proximal_wd`` is False (default) this delegates
        entirely to the base-class implementation.

        When True:
          • Cross-entropy loss is computed as normal.
          • A proximal (anchor) regularisation term is added:
                (weight_decay / 2) * Σ_i ||θ_i − θ_anchor_i||²
            where θ_anchor_i are the weights captured from the initial
            checkpoint *before* any gradient steps.
          • The backward pass is taken on (ce_loss + proximal_loss).
          • The ``"ce"`` metric logged is the *pure* cross-entropy so it
            remains comparable across runs with and without proximal decay.
          • ``proximal_loss`` is accumulated on ``self._proximal_loss_accum``
            (not returned in the dict) so the base ``run_batch`` accumulator
            loop works unchanged.  The value is retrieved by the ``run_batch``
            override below and added to the returned results after super().
        """
        if not getattr(self.config, "gp_proximal_wd", False):
            return super().run_micro_batch(micro_batch, micro_batch_idx, num_micro_batches)

        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)
        micro_d = micro_d.to(self.config.device)

        y_train = micro_y[:, :train_size]
        y_test  = micro_y[:, train_size:]

        if self.ddp:
            self.model.require_backward_grad_sync = (
                micro_batch_idx == num_micro_batches - 1
            )

        with self.amp_ctx:
            pred = self.model(micro_X, y_train, micro_d)  # (B, test_size, n_cls)
            pred = pred.flatten(end_dim=-2)
            true = y_test.long().flatten()
            ce_loss = F.cross_entropy(pred, true)

            # Proximal term: (λ/2) * Σ_i ||θ_i − θ_anchor_i||²
            trainable_params = [
                p for p in self.raw_model.parameters() if p.requires_grad
            ]
            prox = torch.stack(
                [((p - a) ** 2).sum()
                 for p, a in zip(trainable_params, self._anchor_params)]
            ).sum()
            proximal_loss = (self.config.weight_decay / 2.0) * prox
            loss = ce_loss + proximal_loss

        scaled_total = loss / num_micro_batches
        self.scaler.scale(scaled_total).backward()

        with torch.no_grad():
            # Accumulate proximal_loss on self so the base run_batch loop
            # (which only knows about "ce" and "accuracy") doesn't KeyError.
            self._proximal_loss_accum += proximal_loss.item() / num_micro_batches
            micro_results = {
                "ce": ce_loss.item() / num_micro_batches,
                "accuracy": (
                    (pred.argmax(dim=1) == true).float().mean().item()
                    / num_micro_batches
                ),
            }
        return micro_results

    # ------------------------------------------------------------------
    def run_batch(self, batch):
        """
        Thin wrapper around the base ``run_batch`` that, when proximal weight
        decay is active, resets the per-step proximal accumulator before the
        micro-batch loop and appends the accumulated value to the returned
        results dict afterwards.
        """
        if not getattr(self.config, "gp_proximal_wd", False):
            return super().run_batch(batch)

        self._proximal_loss_accum = 0.0
        results = super().run_batch(batch)
        results["proximal_loss"] = self._proximal_loss_accum
        return results

    # ------------------------------------------------------------------
    def configure_prior(self):
        """
        Replace the standard PriorDataset with GPPriorDataset.

        Reads GP-specific settings from self.config (see module docstring).
        """
        data_dir        = self.config.gp_data_dir
        train_manifest  = getattr(self.config, "gp_train_manifest", "train_manifest.csv")
        max_features    = getattr(self.config, "gp_max_features",   None)
        cache_pickles   = getattr(self.config, "gp_cache_pickles",  True)

        dataset = GPPriorDataset(
            data_dir=data_dir,
            batch_size=self.config.batch_size,
            max_features=max_features,
            manifest_filename=train_manifest,
            seed=self.config.np_seed,
            cache_pickles=cache_pickles,
        )

        # Store max_features on config so the eval batch uses the same value
        self.config.gp_max_features = dataset.max_features

        if self.master_process:
            print(dataset)

        self.dataloader = DataLoader(
            dataset,
            batch_size=None,           # batching handled inside GPPriorDataset
            shuffle=False,
            num_workers=1,
            prefetch_factor=4,
            pin_memory=True if self.config.prior_device == "cpu" else False,
            pin_memory_device=(
                self.config.device if self.config.prior_device == "cpu" else ""
            ),
        )

    # ------------------------------------------------------------------
    def _prepare_eval_data(self):
        """
        Pre-generate the fixed evaluation batch from eval_manifest.csv.

        Called once at the start of train().  Stored as self._eval_tasks:
        a list of (X_i, y_i, n_feat, seq_len, train_size) tuples.
        """
        data_dir      = self.config.gp_data_dir
        eval_manifest = getattr(self.config, "gp_eval_manifest",  "eval_manifest.csv")
        n_eval        = getattr(self.config, "gp_n_eval",         10)
        eval_seed     = getattr(self.config, "gp_eval_seed",      0)
        max_features  = self.config.gp_max_features  # already set by configure_prior

        if self.master_process:
            print(
                f"Preparing {n_eval} eval tasks from "
                f"{os.path.join(data_dir, eval_manifest)} ..."
            )

        self._eval_tasks = prepare_eval_batch(
            data_dir=data_dir,
            n_eval=n_eval,
            eval_manifest_filename=eval_manifest,
            max_features=max_features,
            seed=eval_seed,
        )

        if self.master_process:
            print(f"  → {len(self._eval_tasks)} eval tasks ready.")

    # ------------------------------------------------------------------
    def _run_eval(self) -> dict:
        """
        Evaluate on the pre-generated eval tasks.

        Returns
        -------
        dict with keys "eval_ce", "eval_accuracy"
        """
        raw_model = self.raw_model
        raw_model.eval()

        total_ce  = 0.0
        total_acc = 0.0
        n = len(self._eval_tasks)

        for X_i, y_i, n_feat, seq_len, train_size in self._eval_tasks:
            ce, acc = _eval_single_task(
                model=raw_model,
                X_i=X_i,
                y_i=y_i,
                n_feat=n_feat,
                train_size=train_size,
                device=self.config.device,
            )
            total_ce  += ce
            total_acc += acc

        raw_model.train()
        return {"eval_ce": total_ce / n, "eval_accuracy": total_acc / n}

    # ------------------------------------------------------------------
    def _eval_log_and_checkpoint(self, log_step: int, train_metrics: Optional[dict] = None) -> dict:
        """
        Run eval, optionally merge train metrics, log to wandb, and save best-eval checkpoint.

        Parameters
        ----------
        log_step : int
            Step index used for wandb logging and checkpoint naming.
        train_metrics : dict or None
            If provided (normal training step), stored as the latest train values
            and merged into the wandb log dict.
            If None (eval-only call – pre/post training), the last known train
            values are reused so wandb receives complete rows.  If no train step
            has been taken yet the train keys are omitted entirely.

        Returns
        -------
        eval_results : dict with keys "eval_ce" and "eval_accuracy"
        """
        eval_results = self._run_eval()

        # Keep last train metrics up to date
        if train_metrics is not None:
            self._last_train_metrics = {
                "train_ce":       train_metrics["ce"],
                "train_accuracy": train_metrics["accuracy"],
                "train_time":     train_metrics["train_time"],
            }

        # ---- Best-eval checkpoint -----------------------------------------
        if eval_results["eval_accuracy"] > self._best_eval_acc:
            self._best_eval_acc = eval_results["eval_accuracy"]
            new_best_name = f"best-eval-step-{log_step}.ckpt"
            if self._best_eval_ckpt_path and os.path.isfile(self._best_eval_ckpt_path):
                os.remove(self._best_eval_ckpt_path)
            self.save_checkpoint(name=new_best_name)
            self._best_eval_ckpt_path = os.path.join(
                self.config.checkpoint_dir, new_best_name
            )
            print(
                f"\n  [Step {log_step}] New best eval accuracy: "
                f"{self._best_eval_acc:.4f}  → saved {new_best_name}"
            )

        # ---- Wandb logging ------------------------------------------------
        if self.wandb_run is not None:
            try:
                current_lr = self.scheduler.get_last_lr()[0]
            except AttributeError:
                current_lr = self.optimizer.param_groups[0]["lr"]

            log_dict: dict = {}
            if self._last_train_metrics is not None:
                log_dict.update(self._last_train_metrics)
            log_dict.update({**eval_results, "lr": current_lr})
            wandb.log(log_dict, step=log_step)

        return eval_results

    # ------------------------------------------------------------------
    @ddp_cleanup
    def train(self):
        """
        Training loop with per-step evaluation.

        Overrides Trainer.train().  After every gradient-update step:
          • Runs _run_eval() on the fixed eval batch.
          • Logs eval metrics to wandb alongside train metrics.
          • Saves a 'best-eval.ckpt' checkpoint whenever eval accuracy improves.
        """
        # Pre-generate fixed eval batch once
        self._prepare_eval_data()

        self._best_eval_acc: float = -1.0
        self._best_eval_ckpt_path: Optional[str] = None
        self._last_train_metrics: Optional[dict] = None

        if self.master_process:
            step_progress = tqdm(
                range(self.curr_step, self.config.max_steps),
                desc="Step",
                leave=True,
            )
        else:
            step_progress = range(self.curr_step, self.config.max_steps)

        dataloader = iter(self.dataloader)

        for step in step_progress:
            # ---- Pre-step eval (before this step's gradient update) ---------
            if self.master_process:
                eval_results = self._eval_log_and_checkpoint(log_step=step)

            # ---- Fetch next training batch ----------------------------------
            batch = next(dataloader)

            # ---- Gradient update (timed) ------------------------------------
            with Timer() as train_timer:
                train_results = self.run_batch(batch)
            torch.cuda.empty_cache()

            self.curr_step = step + 1

            if self.master_process:
                # Store train metrics for reuse in future eval-only log rows
                self._last_train_metrics = {
                    "train_ce":       train_results["ce"],
                    "train_accuracy": train_results["accuracy"],
                    "train_time":     train_timer.elapsed,
                }
                if "proximal_loss" in train_results:
                    self._last_train_metrics["train_proximal_loss"] = train_results["proximal_loss"]

                # Update progress bar
                step_progress.set_postfix(
                    **{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in {
                           **self._last_train_metrics,
                           **eval_results,
                       }.items()}
                )

                # ---- Periodic checkpoint management ------------------------
                is_temp_save = self.curr_step % self.config.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.save_perm_every == 0

                if is_temp_save or is_perm_save:
                    ckpt_name = f"step-{self.curr_step}.ckpt"
                    self.save_checkpoint(name=ckpt_name)
                    if is_temp_save and not is_perm_save and self.config.max_checkpoints > 0:
                        self.manage_checkpoint()
                    if is_perm_save:
                        print(f"  [Step {self.curr_step}] Permanent checkpoint saved: {ckpt_name}")

        # ---- Post-training eval (after all steps complete) ------------------
        if self.master_process:
            print(f"\n[Post-training eval at step {self.curr_step}]")
            self._eval_log_and_checkpoint(log_step=self.curr_step)
