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

"""SFCQR calibration over a pool of independently-trained ensemble batches.

Companion to `train_ensemble_pool.py`. This script never trains anything: it
only restores checkpoints produced by that script, concatenates them into one
big ensemble "pool" per (alpha, lo/hi), and repeatedly subsamples `n_ens_sub`
members from that pool to compute the SFCQR calibrated thresholds `Q_p`. This
gives a distribution (mean + error bars) of `Q_p`, and downstream coverage,
as a function of ensemble size -- without ever retraining.

It always evaluates on the *same* `cal_idx` saved by `train_ensemble_pool.py`
(`split.npz` in the shared workdir root), and never touches `train_idx`
except to compute the fixed per-channel floor `delta`.

Usage (example):
  python calibrate_sfcqr.py --config configs/config.py:sfcqrclaude_smallnn_beltrami_adamw \
      --batch_n_ens 8 --n_ens_sub_list 1,2,4,8 --n_repeats 20
"""

import functools
import glob
import os
import re
import sys

from absl import app, flags, logging
from ml_collections import config_flags
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.append("./")

from data.ks3dDataGeneration import *
from data.beltramiDataGeneration import *
from model.unet3d import UNet3d
from train.models import QuantileEnsembleDAModel
from train.trainers import BeltramiTrainer
from train.train_states import TrainState
from train.optimizer import *

logging.set_verbosity(logging.ERROR)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/config.py")
flags.DEFINE_integer(
    "batch_n_ens", 8, "Number of ensemble members per training batch. Must "
    "match the value used with `train_ensemble_pool.py`."
)
flags.DEFINE_string(
    "n_ens_sub_list", "1,2,4,8",
    "Comma-separated sub-ensemble sizes to evaluate coverage/Q_p for."
)
flags.DEFINE_integer(
    "n_repeats", 20,
    "Number of random subsamples to draw per sub-ensemble size, for error "
    "bars over the choice of ensemble members."
)
flags.DEFINE_integer(
    "subsample_seed", 0, "Seed controlling which members are drawn in each "
    "repeat (independent of the training/data seed)."
)
flags.DEFINE_integer(
    "predict_chunk_size", 4, "Number of pool members to run through the "
    "U-Net at once when computing calibration-set predictions. Lower this "
    "if you hit an out-of-memory error; raise it (up to the pool size) for "
    "speed if you have memory to spare."
)

SPLIT_FILENAME = "split.npz"
BATCH_DIR_RE = re.compile(r"batch_(\d+)$")


def load_split(workdir_root: str):
  split_path = os.path.join(workdir_root, SPLIT_FILENAME)
  if not os.path.exists(split_path):
    raise FileNotFoundError(
        f"No split file at {split_path}. Run train_ensemble_pool.py first."
    )
  with np.load(split_path) as data:
    return np.array(data["train_idx"]), np.array(data["cal_idx"])


def discover_batch_dirs(label_dir: str) -> list[str]:
  """Finds all `batch_<i>` dirs under `label_dir`, sorted by index."""
  candidates = glob.glob(os.path.join(label_dir, "batch_*"))
  indexed = []
  for c in candidates:
    m = BATCH_DIR_RE.search(c)
    if m:
      indexed.append((int(m.group(1)), c))
  indexed.sort(key=lambda t: t[0])
  return [c for _, c in indexed]


def build_optimizer(cfg):
  if cfg.optimizer == "adamw":
    return optax.adamw(learning_rate=cfg.learning_rate)
  elif cfg.optimizer == "asgld":
    return asgld(
        learning_rate=cfg.learning_rate, tau=cfg.optim_noise, mu_dtype=jnp.bfloat16
    )
  elif cfg.optimizer == "adamsgld":
    return adam_sgld(learning_rate=cfg.learning_rate)
  elif cfg.optimizer == "adamwnoise":
    return optax.chain(
        optax.adamw(learning_rate=cfg.learning_rate, weight_decay=0.0),
        add_langevin_noise(learning_rate=cfg.learning_rate, seed=123),
    )
  raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def restore_pool_variables(
    batch_dirs: list[str],
    unet,
    input_shape: tuple[int, ...],
    tau: float,
    batch_n_ens: int,
    optimizer_cfg,
):
  """Restores and concatenates all batches' model_variables along axis 0.

  Returns a pytree shaped like `{"params": ..., <mutables>: ...}` with a
  leading ensemble axis of size `batch_n_ens * len(batch_dirs)`.
  """
  if not batch_dirs:
    raise ValueError("No batch checkpoints found.")

  da_model = QuantileEnsembleDAModel(
      model=unet, input_shape=input_shape, n_ens=batch_n_ens, tau=tau
  )
  tx = build_optimizer(optimizer_cfg)
  # Only used to get a reference pytree structure/shape for restoring; its
  # random init is discarded as soon as the checkpoint is restored over it.
  ref_trainer = BeltramiTrainer(
      model=da_model, optimizer=tx, rng=jax.random.PRNGKey(0)
  )
  ref_state = ref_trainer.train_state

  all_variables = []
  for batch_dir in batch_dirs:
    ckpt_dir = os.path.join(batch_dir, "checkpoints")
    restored_state = TrainState.restore_from_orbax_ckpt(
        ckpt_dir, field="default", ref_state=ref_state
    )
    all_variables.append(restored_state.model_variables)

  return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *all_variables)


@functools.lru_cache(maxsize=None)
def _build_chunked_apply(unet):
  """Builds (once per `unet`) a jitted, vmapped-over-ensemble-axis apply fn.

  `obs`/`dt` are real (traced) arguments here, not closed-over Python values,
  and the jitted function object is cached and reused across every call --
  both are required for JAX to hit its compilation cache. Building a fresh
  `jax.jit(jax.vmap(...))` per call (as an earlier version of this code did)
  or closing over `obs`/`dt` forces XLA to recompile the whole U-Net on
  *every single call*, which is what made prior runs of this pipeline take
  hours instead of minutes.
  """
  def single_apply(p, m, obs, dt):
    return unet.apply({"params": p, **m}, obs, time=dt, is_training=False)

  return jax.jit(jax.vmap(single_apply, in_axes=(0, 0, None, None)))


def predict_pool(unet, variables, observations, times, member_chunk_size=4):
  """Runs every member of a pool through `unet`, chunked over the ensemble axis.

  `variables` has a leading ensemble axis on every leaf (as produced by
  `restore_pool_variables`). Returns predictions of shape
  [n_pool, N, r1, r2, r3, channels].

  A single `jax.vmap` over the whole pool holds every member's activations
  in memory at once, which is fine for the small `n_ens` (<=16) the original
  single-run training used but OOMs once the pool grows to tens of members.
  We chunk over the ensemble axis with a plain Python loop + `jax.vmap` per
  chunk (concatenating the results) rather than `jax.lax.map`/`lax.scan`:
  scanning a full 3D U-Net body compiles very slowly and can itself exhaust
  host memory during compilation.
  """
  params = variables["params"]
  mutables = {k: v for k, v in variables.items() if k != "params"}
  batch_size = observations.shape[0]
  dt = jnp.full((batch_size,), times[1] - times[0])

  chunked_apply = _build_chunked_apply(unet)

  n_pool = jax.tree.leaves(params)[0].shape[0]
  chunk = min(member_chunk_size, n_pool)

  outputs = []
  for start in range(0, n_pool, chunk):
    end = min(start + chunk, n_pool)
    p_chunk = jax.tree.map(lambda x: x[start:end], params)
    m_chunk = jax.tree.map(lambda x: x[start:end], mutables)
    outputs.append(chunked_apply(p_chunk, m_chunk, observations, dt))

  return jnp.concatenate(outputs, axis=0)


def flat_resids_from_reps(
    y_reps_lo: jax.Array,
    y_reps_hi: jax.Array,
    y_true: jax.Array,
    delta: jax.Array,
    variant: str,
):
  """SFCQR normalised-residual computation, verbatim logic from
  `main_sfcqr_claude.py` (consensus predictor + local scale + residuals).
  """
  n_ens, n_cal, r1, r2, r3, n_channels = y_reps_lo.shape
  m_points = r1 * r2 * r3

  qbar_lo = jnp.mean(y_reps_lo, axis=0)
  qbar_hi = jnp.mean(y_reps_hi, axis=0)
  w = qbar_hi - qbar_lo

  std_lo = jnp.std(y_reps_lo, axis=0)
  std_hi = jnp.std(y_reps_hi, axis=0)
  d = 0.5 * (std_lo + std_hi)

  if variant == "A":
    sigma = jnp.maximum(w, 0.0) + delta
  elif variant == "B":
    sigma = jnp.maximum(w, 0.0) + d + delta
  else:
    raise ValueError(f"Unknown scale variant: {variant}")

  raw_resid = jnp.maximum(qbar_lo - y_true, y_true - qbar_hi)
  resids = raw_resid / sigma
  return resids.reshape(n_cal, m_points, n_channels)


def calibrate_from_flat_resids(flat_resids: jax.Array, alpha: float, beta: float):
  n_cal, m_points, n_channels = flat_resids.shape
  k = int(np.ceil((1 - alpha) * m_points))
  S_p_i = jnp.sort(flat_resids, axis=1)[:, k - 1, :]

  k_beta = int(np.ceil((n_cal + 1) * (1 - beta)))
  sorted_S = jnp.sort(S_p_i, axis=0)
  if k_beta > n_cal:
    return jnp.full((n_channels,), jnp.inf)
  return sorted_S[k_beta - 1, :]


def main(_):
  cfg = FLAGS.config
  workdir_root = os.path.abspath(cfg.workdir)
  batch_n_ens = FLAGS.batch_n_ens
  n_ens_sub_list = [int(x) for x in FLAGS.n_ens_sub_list.split(",")]
  n_repeats = FLAGS.n_repeats
  sub_rng = np.random.default_rng(FLAGS.subsample_seed)

  if cfg.type == "beltrami":
    out_dim = 4
    gt, obs, coords, times = generate_beltrami_da_data(
        spatial_res=cfg.res,
        time_steps=cfg.time_steps,
        dt=cfg.dt,
        a=1.0,
        d=1.0,
        noise_std=cfg.noise_std,
    )
  elif cfg.type == "ks3d":
    out_dim = 1
    gt, obs, times = generate_ks3d_da_data(
        spatial_res=cfg.res, time_steps=cfg.time_steps, noise_std=cfg.noise_std
    )

  train_idx, cal_idx = load_split(workdir_root)
  y_train_true = gt[train_idx + 2]
  n_channels = y_train_true.shape[-1]
  delta = 1e-6 * jnp.std(y_train_true.reshape(-1, n_channels), axis=0)
  delta = delta.reshape(1, 1, 1, 1, n_channels)
  y_true = gt[cal_idx + 2]

  unet = UNet3d(
      out_channels=out_dim,
      time_cond=True,
      num_channels=cfg.model.num_channels,
      use_spatial_attention=cfg.model.use_spatial_attention,
      downsample_ratio=cfg.model.downsample_ratio,
  )
  input_shape = (cfg.batch_size, cfg.res, cfg.res, cfg.res, out_dim)

  for alpha in cfg.alpha_list:
    experiment_dir = os.path.join(workdir_root, f"alpha_{alpha}")
    quantiles = {"lo": alpha / 2, "hi": 1 - alpha / 2}

    pool_preds = {}
    for label, tau in quantiles.items():
      label_dir = os.path.join(experiment_dir, label)
      batch_dirs = discover_batch_dirs(label_dir)
      logging.info(f"alpha={alpha} {label}: found {len(batch_dirs)} batches")

      pool_vars = restore_pool_variables(
          batch_dirs, unet, input_shape, tau, batch_n_ens, cfg
      )
      pool_preds[label] = predict_pool(
          unet, pool_vars, obs[cal_idx], times,
          member_chunk_size=FLAGS.predict_chunk_size,
      )

    n_pool = pool_preds["lo"].shape[0]
    assert pool_preds["hi"].shape[0] == n_pool

    results = {}
    for n_ens_sub in n_ens_sub_list:
      if n_ens_sub > n_pool:
        logging.warning(
            f"Skipping n_ens_sub={n_ens_sub} > pool size {n_pool} for alpha={alpha}"
        )
        continue

      qp_draws = []
      lo_idx_draws = []
      hi_idx_draws = []
      for _ in range(n_repeats):
        lo_idx = sub_rng.choice(n_pool, size=n_ens_sub, replace=False)
        hi_idx = sub_rng.choice(n_pool, size=n_ens_sub, replace=False)
        y_reps_lo = pool_preds["lo"][lo_idx]
        y_reps_hi = pool_preds["hi"][hi_idx]

        flat_resids = flat_resids_from_reps(
            y_reps_lo, y_reps_hi, y_true, delta, cfg.variant
        )
        qp = calibrate_from_flat_resids(flat_resids, alpha, cfg.beta)
        qp_draws.append(np.array(qp))
        lo_idx_draws.append(lo_idx)
        hi_idx_draws.append(hi_idx)

      qp_draws = np.stack(qp_draws, axis=0)  # [n_repeats, n_channels]
      lo_idx_draws = np.stack(lo_idx_draws, axis=0)  # [n_repeats, n_ens_sub]
      hi_idx_draws = np.stack(hi_idx_draws, axis=0)  # [n_repeats, n_ens_sub]
      results[n_ens_sub] = (qp_draws, lo_idx_draws, hi_idx_draws)

      print(
          f"alpha={alpha} n_ens_sub={n_ens_sub}: "
          f"Q_p mean={qp_draws.mean(axis=0)} std={qp_draws.std(axis=0)}"
      )

    save_path = os.path.join(experiment_dir, "qp_by_ensemble_size.npz")
    save_arrays = {}
    for n_ens_sub, (qp_draws, lo_idx_draws, hi_idx_draws) in results.items():
      save_arrays[f"n_ens_{n_ens_sub}"] = qp_draws
      save_arrays[f"n_ens_{n_ens_sub}_lo_idx"] = lo_idx_draws
      save_arrays[f"n_ens_{n_ens_sub}_hi_idx"] = hi_idx_draws
    np.savez(save_path, **save_arrays)
    logging.info(
        f"Saved Q_p distributions (with the member indices used for each "
        f"repeat, so downstream analysis can reuse the exact same "
        f"subsample) to {save_path}"
    )


if __name__ == "__main__":
  app.run(main)
