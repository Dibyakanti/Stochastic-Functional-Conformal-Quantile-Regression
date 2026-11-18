# Copyright 2025 The swirl_dynamics Authors.
# Modification by Dibyakanti Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains a *pool* of ensemble members in independent batches.

This is a companion to `main_sfcqr_claude.py` that intentionally does NOT
perform any SFCQR calibration. It only ever touches `train_idx`, never
`cal_idx`. Splitting training out this way lets you accumulate an arbitrarily
large pool of ensemble members over many independent runs (e.g. run this
script 10 times with `--batch_idx 0..9` and `n_ens=8` to get an 80-member
pool), which `calibrate_sfcqr.py` can later subsample from -- with
repetition -- to get error bars on coverage as a function of ensemble size,
without ever retraining anything.

Usage (example):
  python train_ensemble_pool.py --config configs/config.py:sfcqr_smallnn_beltrami_adamw \
      --batch_idx 0 --batch_n_ens 8

Run this with `batch_idx = 0, 1, ..., n_batches - 1` (each as a separate
job/invocation) to build up the pool. `batch_idx` only controls which slice
of the RNG stream each batch's members are drawn from -- it does not change
`train_idx`/`cal_idx`, which are fixed once and shared by every batch and by
`calibrate_sfcqr.py`.
"""

from itertools import product
from collections.abc import Iterable, Sequence
from clu import metric_writers
from typing import Any

import os
from absl import app, flags, logging
from ml_collections import config_flags
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import sys
import utils

sys.path.append("./")

import train.callbacks as cb
from train import trainers
# import main_sfcqr_claude as sfcqr  # reuse `run()` and data-generation glue
from data.ks3dDataGeneration import *
from data.beltramiDataGeneration import *
from data.challengeDataLoading import *
from utils.data_utils import *
from model.unet3d import UNet3d
from train.models import QuantileEnsembleDAModel
from train.trainers import BeltramiTrainer
from train.optimizer import *

logging.set_verbosity(logging.ERROR)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/config.py")
flags.DEFINE_integer(
    "batch_idx", 0, "Index of this ensemble batch (0-indexed). Determines "
    "the RNG slice used to initialize this batch's members; does not affect "
    "the train/cal split."
)
flags.DEFINE_integer(
    "batch_n_ens", 8, "Number of ensemble members to train jointly in this "
    "batch (all trained with one vmapped train_state, as in "
    "QuantileEnsembleDAModel)."
)

SPLIT_FILENAME = "split.npz"


def get_or_create_split(workdir_root: str, cfg, num_samples: int):
  """Loads the fixed train/cal split, creating it once if absent.

  This file is the single source of truth for `train_idx`/`cal_idx` shared by
  every training batch and by `calibrate_sfcqr.py`. It is intentionally
  created only if missing so that re-running training batches (or adding new
  ones later) can never accidentally change which samples are held out for
  calibration.
  """
  split_path = os.path.join(workdir_root, SPLIT_FILENAME)
  if os.path.exists(split_path):
    with np.load(split_path) as data:
      train_idx, cal_idx = data["train_idx"], data["cal_idx"]
    logging.info(f"Loaded existing split from {split_path}")
    return train_idx, cal_idx

  idx = jax.random.permutation(jax.random.PRNGKey(cfg.seed), num_samples - 2)
  idx = idx[idx % 2 != 0]
  train_idx, cal_idx = idx[: num_samples // 4], idx[num_samples // 4 : -2]
  train_idx, cal_idx = np.array(train_idx), np.array(cal_idx)

  os.makedirs(workdir_root, exist_ok=True)
  np.savez(split_path, train_idx=train_idx, cal_idx=cal_idx)
  logging.info(f"Created and saved new split to {split_path}")
  return train_idx, cal_idx

def run(
    *,
    train_dataloader: Iterable[Any],
    trainer: trainers.BaseTrainer,
    workdir: str,
    # training configs
    total_train_steps: int,
    metric_aggregation_steps: int = 50,
    # evaluation configs
    eval_dataloader: Iterable[Any] | None = None,
    eval_every_steps: int = 100,
    num_batches_per_eval: int = 10,
    run_sanity_eval_batch: bool = True,
    # other configs
    metric_writer: metric_writers.MultiWriter | None = None,
    callbacks: Sequence[cb.Callback] = (),
) -> None:
  """Runs trainer for a training task.

  This function runs a trainer in batches of "metric aggregation" steps, where
  the step-wise metrics obtained within the same batch are aggregated
  (i.e. by computing the average and/or std based on the metric defined in
  the trainer class). The aggregated metrics are then automatically saved to a
  tensorflow event file in `workdir`. Evaluation runs periodically, i.e. once
  every `eval_every_steps` steps, if an eval dataloader is provided.

  Args:
    train_dataloader: A dataloader emitting training data in batches.
    trainer: A trainer object hosting the train and eval logic.
    workdir: The working directory where results (e.g. train & eval metrics) and
      progress (e.g. checkpoints) are saved.
    total_train_steps: Total number of training steps to run.
    metric_aggregation_steps: The trainer runs this number of steps at a time,
      after which training metrics are aggregated and logged.
    eval_dataloader: An evaluation dataloader (optional). If set to `None`, no
      evaluation will run.
    eval_every_steps: The period, in number of train steps, at which evaluation
      runs. Must be an integer multiple of `metric_aggregation_steps`.
    num_batches_per_eval: The number of batches to step through every time
      evaluation is run (resulting metrics are aggregated).
    run_sanity_eval_batch: Whether to step through sanity check eval batch
      before training starts. This helps expose runtime issues early, without
      having to wait until evaluation is first triggered (i.e. after
      `eval_every_steps`).
    metric_writer: A metric writer that writes scalar metrics to disc. It is
      also accessible to callbacks for custom writing in other formats.
    callbacks: A sequence of self-contained programs executing non-essential
      logic (e.g. checkpoint saving, logging, timing, profiling etc.).
  """
  if not os.path.exists(workdir):
    os.makedirs(workdir)

  train_iter = iter(train_dataloader)
  eval_iter = None
  run_evaluation = eval_dataloader is not None
  if run_evaluation:
    if eval_every_steps % metric_aggregation_steps != 0:
      raise ValueError(
          f"`eval_every_steps` ({eval_every_steps}) "
          "must be an integer multiple of "
          "`metric_aggregation_steps` ({metric_aggregation_steps})"
      )
    eval_iter = iter(eval_dataloader)
    if run_sanity_eval_batch:
      trainer.eval(eval_iter, num_steps=1)

  if metric_writer is None:
    metric_writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() > 0
    )

  for callback in callbacks:
    callback.metric_writer = metric_writer
    callback.on_train_begin(trainer)

  cur_step = trainer.train_state.int_step
  while cur_step < total_train_steps:
    for callback in callbacks:
      callback.on_train_batches_begin(trainer)

    num_steps = min(total_train_steps - cur_step, metric_aggregation_steps)
    train_metrics = trainer.train(train_iter, num_steps).compute()
    cur_step += num_steps
    metric_writer.write_scalars(cur_step, train_metrics)

    # At train/eval batch end, callbacks are called in reverse order so that
    # they are last-in-first-out, loosely resembling nested python contexts.
    for callback in reversed(callbacks):
      callback.on_train_batches_end(trainer, train_metrics)

    if run_evaluation:
      if cur_step == total_train_steps or cur_step % eval_every_steps == 0:
        for callback in callbacks:
          callback.on_eval_batches_begin(trainer)

        assert eval_iter is not None
        eval_metrics = trainer.eval(eval_iter, num_batches_per_eval).compute()
        eval_metrics_to_log = {
            k: v for k, v in eval_metrics.items() if utils.is_scalar(v)
        }
        metric_writer.write_scalars(cur_step, eval_metrics_to_log)

        for callback in reversed(callbacks):
          callback.on_eval_batches_end(trainer, eval_metrics)

  for callback in reversed(callbacks):
    callback.on_train_end(trainer)

  metric_writer.flush()

def main(_):
  cfg = FLAGS.config
  batch_idx = FLAGS.batch_idx
  batch_n_ens = FLAGS.batch_n_ens
  workdir_root = os.path.abspath(cfg.workdir)

  # --- Data Generation (identical to main_sfcqr_claude.py) ---
  if cfg.type == "beltrami":
    out_dim = 4
    gt, obs, coords, times = generate_beltrami_da_data(
        spatial_res=cfg.res,
        time_steps=cfg.time_steps,
        dt=cfg.dt,
        a=1.0,
        d=1.0,
        noise_std=cfg.noise_std,
        domain=cfg.get("domain", 1.0),
    )
  elif cfg.type == "ks3d":
    out_dim = 1
    gt, obs, times = generate_ks3d_da_data(
        spatial_res=cfg.res,
        time_steps=cfg.time_steps,
        noise_std=cfg.noise_std,
    )
  elif cfg.type == "challenge2_1":
    out_dim = 2
    gt, obs, times = load_challenge_da_data(
        data_dir=cfg.data_dir,
        time_steps=cfg.time_steps,
        noise_std=cfg.noise_std,
    )
  elif cfg.type == "challenge2_2":
    out_dim = 2
    gt, obs, times = load_challenge2_2_da_data(
        data_dir=cfg.data_dir,
        case=cfg.case,
        time_steps=cfg.time_steps,
        noise_std=cfg.noise_std,
    )

  num_samples = gt.shape[0]
  train_idx, _cal_idx = get_or_create_split(workdir_root, cfg, num_samples)
  # `_cal_idx` is loaded only to keep the split file consistent; this script
  # must never index `gt`/`obs` with it.

  unet = UNet3d(
      out_channels=out_dim,
      time_cond=True,
      num_channels=cfg.model.num_channels,
      use_spatial_attention=cfg.model.use_spatial_attention,
      downsample_ratio=cfg.model.downsample_ratio,
      kernel_size=tuple(cfg.model.get("kernel_size", (3, 3, 3))),
  )

  # Each batch gets an independent RNG slice so its members are fresh draws,
  # not repeats of an earlier batch.
  batch_seed_key = jax.random.fold_in(jax.random.PRNGKey(cfg.seed), batch_idx)

  for alpha in cfg.alpha_list:
    experiment_dir = os.path.join(workdir_root, f"alpha_{alpha}")
    os.makedirs(experiment_dir, exist_ok=True)

    quantiles = {"lo": alpha / 2, "hi": 1 - alpha / 2}
    for label, tau in quantiles.items():
      print(
          f"--- Training batch {batch_idx} ({batch_n_ens} members) for "
          f"alpha={alpha}, {label} quantile (tau={tau}) ---"
      )

      batch_workdir = os.path.join(
          experiment_dir, label, f"batch_{batch_idx}"
      )

      da_model = QuantileEnsembleDAModel(
          model=unet,
          input_shape=(cfg.batch_size, *gt.shape[1:]),
          n_ens=batch_n_ens,
          tau=tau,
      )

      if cfg.optimizer == "adamw":
        tx = optax.adamw(learning_rate=cfg.learning_rate)
      elif cfg.optimizer == "asgld":
        tx = asgld(
            learning_rate=cfg.learning_rate,
            tau=cfg.optim_noise,
            mu_dtype=jnp.bfloat16,
        )
      elif cfg.optimizer == "adamsgld":
        tx = adam_sgld(learning_rate=cfg.learning_rate)
      elif cfg.optimizer == "adamwnoise":
        tx = optax.chain(
            optax.adamw(learning_rate=cfg.learning_rate, weight_decay=0.0),
            add_langevin_noise(learning_rate=cfg.learning_rate, seed=123),
        )

      # Fold the batch index into the trainer's init rng too, so params are
      # independent across batches even if the optimizer/model are identical.
      trainer_rng = jax.random.fold_in(batch_seed_key, hash(label) % (2**31))
      trainer = BeltramiTrainer(model=da_model, optimizer=tx, rng=trainer_rng)

      ckpt_options = ocp.CheckpointManagerOptions(
          save_interval_steps=cfg.total_train_steps, max_to_keep=1
      )
      callbacks = [
          cb.TqdmProgressBar(
              total_train_steps=cfg.total_train_steps,
              train_monitors=["train_loss"],
          ),
          cb.TrainStateCheckpoint(base_dir=batch_workdir, options=ckpt_options),
      ]

      run(
          train_dataloader=da_dataloader(
              train_idx, gt, obs, times, batch_size=cfg.batch_size, mode="train"
          ),
          trainer=trainer,
          workdir=batch_workdir,
          total_train_steps=cfg.total_train_steps,
          metric_aggregation_steps=100,
          callbacks=callbacks,
      )

      logging.info(f"Saved batch {batch_idx} checkpoint to {batch_workdir}")


if __name__ == "__main__":
  app.run(main)
