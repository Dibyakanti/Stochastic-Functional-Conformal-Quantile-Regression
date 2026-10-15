"""Shared helpers for the pooled DeepONet/Beltrami SFCQR pipeline.

Used by `train_beltrami_don_pool.py`, `calibrate_beltrami_don_pool.py` and
`analysis_beltrami_don_pool.py`. Deliberately defines no absl flags so
every script can import it without flag-registration clashes. Checkpoint
saving is shared with the Burgers pipeline (`burgers_pool_utils`).

One DeepONet predicts all four channels (u, v, w, p) at once; every array of
targets/predictions has a trailing channel axis of size 4, and SFCQR gives one
Q_p per channel, as in the UNet pipeline. The branch net is the UViT encoder
(`model/uvit_don.py`) applied to the whole noisy field obs[i].
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax import random

from data.beltramiDataGeneration import generate_beltrami_da_data
from model.uvit_don import UViTDeepONetEnsemble
from utils.burgers_pool_utils import META_FILENAME, PARAMS_DIRNAME

SPLIT_FILENAME = "split.npz"
N_CHANNELS = 4  # u, v, w, p
TARGET_OFFSET = 2  # target is obs[i + 2], as in `da_dataloader`


def load_beltrami_data(cfg):
  """Returns gt, obs (T, R, R, R, 4) and the grid coords (R**3, 3)."""
  gt, obs, coords, _ = generate_beltrami_da_data(
      spatial_res=cfg.res,
      time_steps=cfg.time_steps,
      dt=cfg.dt,
      a=1.0,
      d=1.0,
      noise_std=cfg.noise_std,
  )
  grid = jnp.stack(
      jnp.meshgrid(coords["x"], coords["y"], coords["z"], indexing="ij"), axis=-1
  ).reshape(-1, 3)
  return np.asarray(gt), np.asarray(obs), np.asarray(grid)


def get_or_create_split(workdir_root, cfg, num_samples):
  """Time-index train/cal split, fixed once in `split.npz`.

  Same construction as `train_ensemble_pool.py`, so for the same seed and
  time_steps it picks the same time indices as the UNet runs.
  """
  split_path = os.path.join(workdir_root, SPLIT_FILENAME)
  if os.path.exists(split_path):
    with np.load(split_path) as data:
      return data["train_idx"], data["cal_idx"]

  idx = random.permutation(random.PRNGKey(cfg.seed), num_samples - 2)
  idx = idx[idx % 2 != 0]
  train_idx, cal_idx = idx[: num_samples // 4], idx[num_samples // 4 : -2]
  train_idx, cal_idx = np.array(train_idx), np.array(cal_idx)

  # Write-then-rename so parallel jobs never read a half-written file.
  os.makedirs(workdir_root, exist_ok=True)
  tmp_path = f"{split_path}.tmp{os.getpid()}.npz"
  np.savez(tmp_path, train_idx=train_idx, cal_idx=cal_idx)
  os.replace(tmp_path, split_path)
  return train_idx, cal_idx


def branch_fields(obs, fn_idx):
  """Branch input: the whole noisy field obs[i], (len(fn_idx), X, Y, Z, 4)."""
  return np.float32(obs[np.asarray(fn_idx)])


def time_gap(cfg, n):
  """Branch time input: the gap between input and target, as the UNet gets
  it from `da_dataloader` (TARGET_OFFSET * dt for every function)."""
  return np.full((n,), TARGET_OFFSET * cfg.dt, dtype=np.float32)


def point_indices(n_points, fn_idx, n_grid, seed=0):
  """P random flat grid indices per function, (len(fn_idx), n_points).

  Keyed on the time index itself, so a function gets the same points whichever
  split or batch it appears in.
  """
  def one(i):
    return random.choice(
        random.fold_in(random.PRNGKey(seed), i), n_grid, (n_points,), replace=False
    )
  return np.asarray(jax.vmap(one)(jnp.asarray(fn_idx)))


def function_data(fields, grid, fn_idx, n_points):
  """Per-function trunk points and targets, all channels.

  `fields` supplies the targets (obs for training, gt for calibration/testing),
  read at time index i + TARGET_OFFSET. Returns y (n_fn, n_points, 3) and
  s (n_fn, n_points, 4).
  """
  pts = point_indices(n_points, fn_idx, grid.shape[0])
  targets = fields[np.asarray(fn_idx) + TARGET_OFFSET].reshape(
      len(fn_idx), -1, N_CHANNELS
  )
  s = np.take_along_axis(targets, pts[..., None], axis=1)
  return np.float32(grid[pts]), np.float32(s)


def branch_kwargs(cfg):
  b = cfg.model.branch
  return dict(
      num_channels=tuple(b.num_channels),
      downsample_ratio=tuple(b.downsample_ratio),
      use_spatial_attention=tuple(b.use_spatial_attention),
      num_blocks=b.num_blocks,
      noise_embed_dim=b.noise_embed_dim,
      input_proj_channels=b.input_proj_channels,
      num_heads=b.num_heads,
  )


def build_ensemble(cfg, n_ens, tau, lr, rng_key):
  """trunk_layers[-1] is the latent size p per channel."""
  return UViTDeepONetEnsemble(
      n_ens=n_ens,
      branch_kwargs=branch_kwargs(cfg),
      field_shape=(cfg.res, cfg.res, cfg.res, N_CHANNELS),
      trunk_layers=[3] + list(cfg.model.trunk_layers[1:]),
      lr=lr,
      n_out=N_CHANNELS,
      alpha=tau,
      rng_key=rng_key,
  )


def quantile_dir(workdir_root, alpha, lr, label):
  return os.path.join(workdir_root, f"alpha_{alpha}", f"lr_{lr}", label)


def restore_pool_params(cfg, batch_dirs, lr):
  """Restores every batch and concatenates params along the member axis.

  Each batch's own `meta.json` provides its n_ens, so batches of different
  sizes can be mixed.
  """
  if not batch_dirs:
    raise ValueError("No batch checkpoints found.")
  ckptr = ocp.StandardCheckpointer()
  restored = []
  for _, d in batch_dirs:
    with open(os.path.join(d, META_FILENAME)) as f:
      meta = json.load(f)
    if meta.get("branch") != "uvit" or meta.get("n_out") != N_CHANNELS:
      raise ValueError(f"{d} is not a UViT-branch {N_CHANNELS}-output checkpoint.")
    if meta["res"] != cfg.res:
      raise ValueError(
          f"{d} was trained with res={meta['res']}, but the config has "
          f"res={cfg.res}."
      )
    ref = build_ensemble(cfg, meta["n_ens"], meta["tau"], lr, random.PRNGKey(0))
    restored.append(
        ckptr.restore(os.path.abspath(os.path.join(d, PARAMS_DIRNAME)), ref.params)
    )
  return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *restored)


def make_field_predictor(cfg):
  """Jitted fn: (member_params, fields, time, grid) -> [n_members, n_fn, n_grid, 4].

  Evaluates each function on the whole grid: the branch runs once per field
  and the trunk once per grid point.
  """
  ens = build_ensemble(cfg, 1, 0.5, 1e-3, random.PRNGKey(0))
  return jax.jit(jax.vmap(ens.predict_grid, in_axes=(0, None, None, None)))


def predict_fields(predictor, params, fields, time, grid, chunk=8):
  """Pool predictions [n_pool, n_fn, n_grid, 4], `chunk` members at a time."""
  n_pool = jax.tree.leaves(params)[0].shape[0]
  outs = []
  for start in range(0, n_pool, chunk):
    p = jax.tree.map(lambda x: x[start:start + chunk], params)
    outs.append(np.asarray(predictor(p, fields, time, grid), dtype=np.float32))
  return np.concatenate(outs, axis=0)


def field_targets(fields, fn_idx):
  """fields[i + 2] on the whole grid, [n_fn, n_grid, 4]."""
  t = fields[np.asarray(fn_idx) + TARGET_OFFSET]
  return t.reshape(len(fn_idx), -1, N_CHANNELS).astype(np.float32)


def sfcqr_delta(gt, train_idx):
  """Per-channel SFCQR floor [4] from the TRAINING functions' true fields, as
  the UNet uses."""
  return 1e-6 * field_targets(gt, train_idx).reshape(-1, N_CHANNELS).std(axis=0)


def sfcqr_function_scores(qbar_lo, qbar_hi, sigma, y, alpha):
  """Per function and channel, the ceil((1 - alpha) m)-th smallest normalised
  residual max(qbar_lo - y, y - qbar_hi) / sigma over its m points.

  All inputs [n_fn, m, 4]; returns [n_fn, 4].
  """
  resid = np.maximum(qbar_lo - y, y - qbar_hi) / sigma
  k = int(np.ceil((1 - alpha) * resid.shape[1]))
  return np.sort(resid, axis=1)[:, k - 1, :]


def sfcqr_threshold(S, beta):
  """Per-channel Q_p [4]: the ceil((n + 1)(1 - beta))-th smallest of the n
  function scores S [n, 4] (inf if that index exceeds n)."""
  n = S.shape[0]
  k_beta = int(np.ceil((n + 1) * (1 - beta)))
  if k_beta > n:
    return np.full(S.shape[1], np.inf)
  return np.sort(S, axis=0)[k_beta - 1]


def test_indices(cfg):
  """Rollout inputs used by the UNet `analysis_pool.py`: obs[2 + 2i],
  i < num_unroll_steps, with truth gt[4 + 2i]."""
  return 2 + 2 * np.arange(cfg.num_unroll_steps)


def load_split(workdir_root):
  split_path = os.path.join(workdir_root, SPLIT_FILENAME)
  if not os.path.exists(split_path):
    raise FileNotFoundError(
        f"No split file at {split_path}. Run train_beltrami_don_pool.py first."
    )
  with np.load(split_path) as data:
    return data["train_idx"], data["cal_idx"]
