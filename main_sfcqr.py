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
from itertools import product

import os
from clu import metric_writers
import jax
import optax
import sys
import orbax.checkpoint as ocp

sys.path.append("./")

import train.callbacks as cb
from train import trainers
import utils
from data.ks3dDataGeneration import *
from data.beltramiDataGeneration import *
from utils.data_utils import *
from model.unet3d import UNet3d
from train.models import QuantileEnsembleDAModel
from train.trainers import BeltramiTrainer
from train.optimizer import *

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


def get_predictions(trainer, observations, times):
    """Manual prediction extractor that handles ensemble dimensions."""
    # 1. Prepare inputs
    # Use the actual batch size of the observations provided
    batch_size = observations.shape[0]
    dt = jnp.full((batch_size,), times[1] - times[0])
    
    # 2. Extract variables (params and potentially mutables/batch_stats)
    variables = trainer.train_state.model_variables
    params = variables["params"]
    # Handle mutables if they exist (e.g. Batch Norm state), otherwise empty dict
    mutables = {k: v for k, v in variables.items() if k != "params"}

    # 3. Define a single-pass function for one ensemble member
    def single_model_apply(p, m):
        return trainer.model.model.apply(
            {"params": p, **m},
            observations,
            time=dt,
            is_training=False
        )

    # 4. Use jax.vmap to iterate over the ensemble dimension
    # This maps over the leading axis of params and mutables
    ensemble_preds = jax.vmap(single_model_apply)(params, mutables)
    
    # Shape: [n_ens, N_cal, res, res, res, channels]
    return ensemble_preds


def main(_):
    cfg = FLAGS.config
    workdir_root = os.path.abspath(cfg.workdir)
    
    # --- Data Generation ---
    if cfg.type=="beltrami":
      
      out_dim = 4
      
      gt, obs, coords, times = generate_beltrami_da_data(
        spatial_res=cfg.res, 
        time_steps=cfg.time_steps,
        dt=cfg.dt, 
        a=1.0,
        d=1.0,
        noise_std=cfg.noise_std
      )    
    elif cfg.type=="ks3d":
      
      out_dim = 1
      
      gt, obs, times = generate_ks3d_da_data(
        spatial_res=cfg.res, 
        time_steps=cfg.time_steps,
        noise_std=cfg.noise_std
      )

  
    
    # Split into Train and Calibration
    num_samples = gt.shape[0]
    idx = jax.random.permutation(jax.random.PRNGKey(cfg.seed), num_samples-2)
    idx = idx[idx%2!=0]
    train_idx, cal_idx = idx[:num_samples//4], idx[num_samples//4:-2]

    # --- Architecture ---
    unet = UNet3d(
        out_channels=out_dim,
        time_cond=True,
        num_channels=cfg.model.num_channels,
        use_spatial_attention=cfg.model.use_spatial_attention,
        downsample_ratio = cfg.model.downsample_ratio
    )

    # --- Training Setup ---
    trainers_dict = {}
    
    for alpha, n_ens in product(cfg.alpha_list, cfg.n_ens_list):
      # Create a unique path
      experiment_dir = os.path.join(workdir_root, f"alpha_{alpha}_nens_{n_ens}")
      os.makedirs(experiment_dir, exist_ok=True)
      
      quantiles = {"lo": alpha/2, "hi": 1 - alpha/2}
      for label, tau in quantiles.items():
          print(f"--- Training {label} quantile (tau={tau}) ---")
          
          # Unique workdir for each quantile
          q_workdir = os.path.join(experiment_dir, label)
          
          da_model = QuantileEnsembleDAModel(
              model=unet,
              input_shape=(cfg.batch_size, cfg.res, cfg.res, cfg.res, out_dim),
              n_ens=n_ens,
              tau=tau
          )

          if cfg.optimizer=="adamw":
            tx = optax.adamw(learning_rate=cfg.learning_rate)
          elif cfg.optimizer=="asgld":
            tx = asgld(learning_rate=cfg.learning_rate, tau = cfg.optim_noise, mu_dtype=jnp.bfloat16)
          elif cfg.optimizer=="adamsgld":
            tx = adam_sgld(learning_rate=cfg.learning_rate)
          elif cfg.optimizer=="adamwnoise":
            tx = optax.chain(
                optax.adamw(learning_rate=cfg.learning_rate,weight_decay=0.0),
                add_langevin_noise(learning_rate=cfg.learning_rate,seed=123)
                )
          trainer = BeltramiTrainer(model=da_model, optimizer=tx, rng=jax.random.PRNGKey(cfg.seed))

          ckpt_options = ocp.CheckpointManagerOptions(
              save_interval_steps=cfg.total_train_steps, 
              max_to_keep=1
          )
          
          callbacks = [
              cb.TqdmProgressBar(total_train_steps=cfg.total_train_steps, train_monitors=["train_loss"]),
              cb.TrainStateCheckpoint(base_dir=q_workdir, options=ckpt_options)
          ]

          run(
              train_dataloader=da_dataloader(
                  train_idx, gt, obs, times, batch_size=cfg.batch_size, mode="train"
              ),
              trainer=trainer,
              workdir=q_workdir,
              total_train_steps=cfg.total_train_steps,
              metric_aggregation_steps=50,
              callbacks=callbacks
          )
          trainers_dict[label] = trainer

      # --- Calibration Implementation ---
      m_points = cfg.res**3  # Total spatial points

      # --- Following Algorithm Steps 4-6 (Per-Channel) ---

      # 1. Get raw ensemble predictions
      y_reps_lo = get_predictions(trainers_dict["lo"], obs[cal_idx], times)
      y_reps_hi = get_predictions(trainers_dict["hi"], obs[cal_idx], times)
      y_true = gt[cal_idx+2] # [N_cal, res, res, res, 4]

      # Step 4: Per-function, per-replicate residuals
      # r = max(q_lo - u, u - q_hi)
      resids = jnp.maximum(y_reps_lo - y_true, y_true - y_reps_hi) 

      # Flatten only the spatial dimensions (res*res*res), keeping channels
      # New shape: [n_ens, N_cal, m_points, 4]
      n_ens, n_cal, r1, r2, r3, n_channels = resids.shape
      m_points = r1 * r2 * r3
      flat_resids = resids.reshape(n_ens, n_cal, m_points, n_channels)

      # Compute k-th order statistic (s_p,i,j) for each channel
      k = int(np.ceil((1 - alpha) * m_points))
      # Sort along the spatial dimension (axis 2) and pick k-th largest
      # Shape: [n_ens, N_cal, 4]
      s_p_i_j = jnp.sort(flat_resids, axis=2)[:, :, k - 1, :]

      # Step 5: Symmetric aggregation across replicates
      # Mean across ensemble dimension (axis 0)
      # Shape: [N_cal, 4]
      S_p_i = jnp.mean(s_p_i_j, axis=0)

      # Step 6: Calibrate the thresholds Q_p per channel
      level = (1 - alpha) * (1 + 1 / n_cal)
      # Quantile across the calibration samples (axis 0)
      Q_p = jnp.quantile(S_p_i, q=level, axis=0, method="higher")

      print(f"Calibrated thresholds per channel: {Q_p}")
      
      # Save the thresholds to the root workdir
      save_path = os.path.join(experiment_dir, "calibrated_thresholds.npy")
      with open(save_path, "wb") as f:
          np.save(f, np.array(Q_p))

      logging.info(f"Saved calibrated thresholds to {save_path}")

if __name__ == "__main__":
    app.run(main)