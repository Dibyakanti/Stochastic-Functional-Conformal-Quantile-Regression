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

"""Function that runs training."""

from collections.abc import Iterable, Sequence
from absl import app, flags, logging
from ml_collections import config_flags
from typing import Any

import os
from clu import metric_writers
import jax
import optax
import sys

sys.path.append("./")

import train.callbacks as cb
from train import trainers
import utils
from data.beltramiDataGeneration import *
from model.unet3d import UNet3d
from train.models import DataAssimilationModel
from train.trainers import BeltramiTrainer

logging.set_verbosity(logging.ERROR)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/config.py")

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
    # cfg is now populated by ml_collections based on your EXP_NAME
    cfg = FLAGS.config

    # --- 1. Directory Setup ---
    # Convert relative workdir to absolute to avoid issues on the cluster
    workdir = os.path.abspath(cfg.workdir)
    os.makedirs(workdir, exist_ok=True)
    
    # Google Best Practice: Save a JSON snapshot of the config used for this run
    with open(os.path.join(workdir, "config.json"), "w") as f:
        f.write(cfg.to_json(indent=4))

    # --- 2. Generate the Dataset ---
    # Using cfg.res and cfg.noise_std from your preset
    gt, obs, coords, times = generate_da_data(
        spatial_res=cfg.res, 
        time_steps=cfg.time_steps,
        dt=cfg.dt, 
        a=1.0,
        d=1.0,
        noise_std=cfg.noise_std
    )

    # --- 3. Setup Architecture ---
    unet_config = UNet3d(
        out_channels=4,
        time_cond=True,
        num_channels=cfg.model.num_channels,
        use_spatial_attention=cfg.model.use_spatial_attention
    )

    # --- 4. Update the Model Wrapper ---
    da_model = DataAssimilationModel(
        model=unet_config, 
        input_shape=(cfg.batch_size, cfg.res, cfg.res, cfg.res, 4) 
    )

    # --- 5. Define Optimizer & Trainer ---
    tx = optax.adamw(learning_rate=cfg.learning_rate, weight_decay=1e-5)
    trainer = BeltramiTrainer(
        model=da_model, 
        optimizer=tx, 
        rng=jax.random.PRNGKey(cfg.seed)
    )

    # --- 6. Define Callbacks ---
    callbacks = [
        cb.TqdmProgressBar(
            total_train_steps=cfg.total_train_steps, 
            train_monitors=["train_loss"]
        ),
        cb.TrainStateCheckpoint(base_dir=workdir)
    ]

    # --- 7. Start Training ---
    run(
        train_dataloader=beltrami_da_dataloader(
            gt, obs, times, batch_size=cfg.batch_size, mode="train"
        ),
        trainer=trainer,
        workdir=workdir,
        total_train_steps=cfg.total_train_steps,
        metric_aggregation_steps=500,
        callbacks=callbacks
    )

if __name__ == "__main__":
    app.run(main)