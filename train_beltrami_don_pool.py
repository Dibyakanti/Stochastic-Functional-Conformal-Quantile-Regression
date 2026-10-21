"""Trains one batch of a DeepONet quantile-ensemble pool on Beltrami.

DeepONet counterpart of `train_ensemble_pool.py` (UNet), built the same way as
`train_burgers_pool.py`: trains only, never calibrates. Run it once per
`--batch_idx` to grow the pool.

Each function is a time index i from the training split: the branch (the UViT
encoder of `UNet3d`, see `model/uvit_don.py`) sees the whole noisy field obs[i]
and the time gap, the trunk sees (x, y, z), and the target is obs[i + 2] (all
4 channels). Each step, every member gets its own minibatch of
`fn_batch_size` functions x `pts_per_fn` points: from P_train fixed points per
function, or, with `augment_symmetry`, from the whole grid after a random
cyclic (x, y, z) / (u, v, w) permutation of input and target.
A single DeepONet outputs u, v, w, p together, trained with the quantile loss
averaged over the four outputs. Only `train_idx` from `<workdir>/split.npz` is
used; the calibration time indices are never touched.

Checkpoints:
  <workdir>/alpha_<a>/lr_<lr>/<lo|hi>/batch_<i>/{params/, meta.json, loss_log.npy}

Usage:
  python train_beltrami_don_pool.py \
      --config=configs/beltrami_don_config.py:beltrami_don_sfcqr \
      --batch_idx=0 --batch_n_ens=16
"""

import os
import sys

from absl import app, flags, logging
from jax import config as jax_config
from jax import random
from ml_collections import config_flags

sys.path.append("./")

from utils.beltrami_don_pool_utils import (
    N_CHANNELS,
    build_ensemble,
    get_or_create_split,
    load_beltrami_data,
    make_sampler,
    quantile_dir,
)
from utils.burgers_pool_utils import save_batch

logging.set_verbosity(logging.ERROR)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/beltrami_don_config.py")
flags.DEFINE_integer("batch_idx", 0, "Index of this pool batch.")
flags.DEFINE_integer("batch_n_ens", 8, "Ensemble members trained in this batch.")


def main(_):
  cfg = FLAGS.config
  jax_config.update("jax_enable_x64", True)
  workdir_root = os.path.abspath(cfg.workdir)
  batch_idx, batch_n_ens = FLAGS.batch_idx, FLAGS.batch_n_ens

  _, obs, grid = load_beltrami_data(cfg)
  train_idx, _ = get_or_create_split(workdir_root, cfg, obs.shape[0])

  batch_key = random.fold_in(random.PRNGKey(cfg.seed), batch_idx)

  for alpha in cfg.alpha_list:
    for lr in cfg.lr_list:
      # lo and hi share the init key and minibatch stream, as in
      # train_burgers_pool.py.
      init_key = random.fold_in(batch_key, 0)
      data_key = random.fold_in(batch_key, 1)

      for label, tau in (("lo", alpha / 2), ("hi", 1 - alpha / 2)):
        print(
            f"--- batch {batch_idx} ({batch_n_ens} members): "
            f"alpha={alpha} lr={lr} {label} (tau={tau}) ---"
        )
        sampler = make_sampler(cfg, obs, grid, train_idx, batch_n_ens, data_key)
        ensemble = build_ensemble(cfg, batch_n_ens, tau, lr, init_key)
        ensemble.train(sampler, nIter=cfg.nIter)

        save_batch(
            os.path.join(
                quantile_dir(workdir_root, alpha, lr, label), f"batch_{batch_idx}"
            ),
            ensemble,
            meta={
                "n_ens": batch_n_ens,
                "tau": tau,
                "alpha": alpha,
                "lr": lr,
                "batch_idx": batch_idx,
                "branch": "uvit",
                "res": cfg.res,
                "n_out": N_CHANNELS,
            },
        )


if __name__ == "__main__":
  app.run(main)
