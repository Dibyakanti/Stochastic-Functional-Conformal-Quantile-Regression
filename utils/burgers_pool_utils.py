"""Shared helpers for the pooled DeepONet/Burgers SFCQR pipeline.

Used by `train_burgers_pool.py`, `calibrate_burgers_pool.py` and
`analysis_burgers_pool.py`. Deliberately defines no absl flags so every script
can import it without flag-registration clashes.
"""

import glob
import json
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax import random, vmap

from data.burgersDataGeneration import (
    generate_one_test_data,
    generate_one_training_data,
)
from model.don import DeepONetEnsemble

SPLIT_FILENAME = "split.npz"
META_FILENAME = "meta.json"
PARAMS_DIRNAME = "params"
LOSS_FILENAME = "loss_log.npy"
BATCH_DIR_RE = re.compile(r"batch_(\d+)$")


def _kappas(cfg, n):
  # Same values as np.arange(vmin, vmax, (vmax - vmin) / n), but guaranteed to
  # have exactly n entries (float-step arange can return n + 1).
  return np.linspace(cfg.grid_data.vmin, cfg.grid_data.vmax, n, endpoint=False)


def generate_function_data(cfg):
  """Per-function training/calibration data, as in burgersSCQR.py.

  Returns u_all (N_train, P_train, m), y_all (N_train, P_train, 2),
  s_all (N_train, P_train, 1).
  """
  keys = random.split(random.PRNGKey(0), cfg.N_train)
  return vmap(generate_one_training_data, (0, None, 0))(
      keys, cfg, _kappas(cfg, cfg.N_train)
  )


def generate_test_data(cfg):
  """Flattened test data, as in burgersSCQRAnalysis.py."""
  u_test, y_test, s_test = vmap(generate_one_test_data, (None, 0))(
      cfg, _kappas(cfg, cfg.N_test)
  )
  n_pts = cfg.N_test * cfg.P_test**2
  return (
      jnp.float32(u_test.reshape(n_pts, -1)),
      jnp.float32(y_test.reshape(n_pts, -1)),
      jnp.float32(s_test.reshape(n_pts, -1)),
  )


def get_or_create_split(workdir_root, cfg):
  """Function-level train/cal split, fixed once in `split.npz`.

  Same construction as burgersSCQR.py (half the functions held out for
  calibration via PRNGKey(123)), but persisted so every training batch and
  the calibration script are guaranteed to agree on it.
  """
  split_path = os.path.join(workdir_root, SPLIT_FILENAME)
  if os.path.exists(split_path):
    with np.load(split_path) as data:
      idx_train, idx_cp = data["idx_train"], data["idx_cp"]
    if len(idx_train) + len(idx_cp) != cfg.N_train:
      raise ValueError(
          f"{split_path} was made for N_train={len(idx_train) + len(idx_cp)}, "
          f"but the config has N_train={cfg.N_train}. Use a fresh workdir."
      )
    return idx_train, idx_cp

  n_cal = int(cfg.N_train / 2)
  idx_cp = random.choice(
      random.PRNGKey(123), jnp.arange(cfg.N_train), shape=(n_cal, 1), replace=False
  )[:, 0]
  mask = jnp.ones(cfg.N_train, dtype=bool).at[idx_cp].set(False)
  idx_train = jnp.arange(cfg.N_train)[mask]
  idx_train, idx_cp = np.array(idx_train), np.array(idx_cp)

  # Write-then-rename so parallel array tasks never read a half-written file;
  # they all compute the identical split, so whichever rename lands is fine.
  os.makedirs(workdir_root, exist_ok=True)
  tmp_path = f"{split_path}.tmp{os.getpid()}.npz"
  np.savez(tmp_path, idx_train=idx_train, idx_cp=idx_cp)
  os.replace(tmp_path, split_path)
  return idx_train, idx_cp


def load_split(workdir_root):
  split_path = os.path.join(workdir_root, SPLIT_FILENAME)
  if not os.path.exists(split_path):
    raise FileNotFoundError(
        f"No split file at {split_path}. Run train_burgers_pool.py first."
    )
  with np.load(split_path) as data:
    return data["idx_train"], data["idx_cp"]


def build_ensemble(cfg, n_ens, tau, lr, rng_key):
  return DeepONetEnsemble(
      n_ens=n_ens,
      branch_layers=cfg.model.branch_layers,
      trunk_layers=cfg.model.trunk_layers,
      lr=lr,
      loss_type="quantile",
      alpha=tau,
      rng_key=rng_key,
  )


def quantile_dir(workdir_root, alpha, lr, label):
  return os.path.join(workdir_root, f"alpha_{alpha}", f"lr_{lr}", label)


def save_batch(batch_dir, ensemble, meta):
  params_path = os.path.abspath(os.path.join(batch_dir, PARAMS_DIRNAME))
  if os.path.exists(params_path):
    raise FileExistsError(
        f"{params_path} already exists. Use a new --batch_idx, or delete it "
        "to retrain this batch."
    )
  os.makedirs(batch_dir, exist_ok=True)
  ckptr = ocp.StandardCheckpointer()
  ckptr.save(params_path, ensemble.params)
  ckptr.wait_until_finished()
  np.save(os.path.join(batch_dir, LOSS_FILENAME), np.array(ensemble.loss_don_log))
  with open(os.path.join(batch_dir, META_FILENAME), "w") as f:
    json.dump(meta, f, indent=2)


def discover_batch_dirs(label_dir):
  """Returns [(batch_idx, path)] sorted by batch index."""
  out = []
  for c in glob.glob(os.path.join(label_dir, "batch_*")):
    m = BATCH_DIR_RE.search(c)
    if m and os.path.exists(os.path.join(c, PARAMS_DIRNAME)):
      out.append((int(m.group(1)), c))
  return sorted(out)


def restore_pool_params(cfg, batch_dirs, lr):
  """Restores every batch and concatenates params along the member axis.

  Each batch's own `meta.json` provides its n_ens, so batches of different
  sizes can be mixed and no --batch_n_ens flag has to match training.
  """
  if not batch_dirs:
    raise ValueError("No batch checkpoints found.")
  ckptr = ocp.StandardCheckpointer()
  restored = []
  for _, d in batch_dirs:
    with open(os.path.join(d, META_FILENAME)) as f:
      meta = json.load(f)
    ref = build_ensemble(cfg, meta["n_ens"], meta["tau"], lr, random.PRNGKey(0))
    restored.append(
        ckptr.restore(os.path.abspath(os.path.join(d, PARAMS_DIRNAME)), ref.params)
    )
  return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *restored)


def make_pool_predictor(cfg):
  """Jitted fn: (member_params, u, z) -> predictions, vmapped over members.

  Built once and reused, with params/inputs passed as arguments rather than
  closed over, so XLA compiles once instead of on every call.
  """
  # Architecture only -- tau/lr/key don't affect the forward pass.
  ens = build_ensemble(cfg, 1, 0.5, 1e-3, random.PRNGKey(0))
  return jax.jit(vmap(ens.predict_u, in_axes=(0, None, None)))


def predict_pool(predictor, params, u, z, chunk=8):
  """Pool predictions [n_pool, n_points, 1], computed `chunk` members at a time."""
  n_pool = jax.tree.leaves(params)[0].shape[0]
  outs = []
  for start in range(0, n_pool, chunk):
    p = jax.tree.map(lambda x: x[start:start + chunk], params)
    outs.append(np.asarray(predictor(p, u, z)))
  return np.concatenate(outs, axis=0)


# --- SFCQR, same algorithm as the UNet pipeline (calibrate_sfcqr.py) ---


def sfcqr_delta(s_all, idx_train):
  """Tiny floor so sigma > 0, from the TRAINING functions only."""
  return 1e-6 * float(np.std(np.asarray(s_all[idx_train])))


def sfcqr_consensus(reps_lo, reps_hi, delta, variant):
  """Consensus bounds and local scale from member predictions.

  reps_lo/reps_hi: [k, ..., 1]. Returns (qbar_lo, qbar_hi, sigma), each [..., 1].
  Variant A: sigma = max(w, 0) + delta; variant B adds the mean member std d.
  """
  qbar_lo, qbar_hi = reps_lo.mean(axis=0), reps_hi.mean(axis=0)
  sigma = np.maximum(qbar_hi - qbar_lo, 0.0) + delta
  if variant == "B":
    sigma = sigma + 0.5 * (reps_lo.std(axis=0) + reps_hi.std(axis=0))
  elif variant != "A":
    raise ValueError(f"Unknown scale variant: {variant}")
  return qbar_lo, qbar_hi, sigma


def sfcqr_function_scores(qbar_lo, qbar_hi, sigma, y, alpha):
  """One score per function: the ceil((1 - alpha) m)-th smallest normalised
  residual max(qbar_lo - y, y - qbar_hi) / sigma over its m points.

  All inputs [n_fn, m, 1]; returns [n_fn].
  """
  resid = (np.maximum(qbar_lo - y, y - qbar_hi) / sigma)[..., 0]
  k = int(np.ceil((1 - alpha) * resid.shape[1]))
  return np.sort(resid, axis=1)[:, k - 1]


def sfcqr_threshold(S, beta):
  """Functional-level threshold: the ceil((n + 1)(1 - beta))-th smallest score
  (inf if that index exceeds n)."""
  n = len(S)
  k_beta = int(np.ceil((n + 1) * (1 - beta)))
  if k_beta > n:
    return np.inf
  return float(np.sort(S)[k_beta - 1])
