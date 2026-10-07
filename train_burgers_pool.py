"""Trains one batch of a DeepONet quantile-ensemble pool on Burgers.

Pooled counterpart of `burgersSCQR.py`: trains only, never calibrates. Run it
once per `--batch_idx` to grow the pool; `calibrate_burgers_pool.py` then
subsamples ensembles of any size from it.

Only the training half of the function-level split (`idx_train` in
`<workdir>/split.npz`) is ever used here; the calibration functions are never
touched.

Checkpoints: <workdir>/alpha_<a>/lr_<lr>/<lo|hi>/batch_<i>/{params/, meta.json, loss_log.npy}

"""

import os
import sys

from absl import app, flags, logging
from jax import config as jax_config
from jax import random
import jax.numpy as jnp
from ml_collections import config_flags

sys.path.append("./")

from data.burgersDataGeneration import BatchedDataGenerator, DataGenerator
from utils.burgers_pool_utils import (
    build_ensemble,
    generate_function_data,
    get_or_create_split,
    quantile_dir,
    save_batch,
)

logging.set_verbosity(logging.ERROR)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/burgers_config.py")
flags.DEFINE_integer("batch_idx", 0, "Index of this pool batch.")
flags.DEFINE_integer("batch_n_ens", 8, "Ensemble members trained in this batch.")


def main(_):
  cfg = FLAGS.config
  jax_config.update("jax_enable_x64", True)
  workdir_root = os.path.abspath(cfg.workdir)
  batch_idx, batch_n_ens = FLAGS.batch_idx, FLAGS.batch_n_ens

  u_all, y_all, s_all = generate_function_data(cfg)
  idx_train, _ = get_or_create_split(workdir_root, cfg)

  n_pts = len(idx_train) * cfg.P_train
  u_train = jnp.float32(u_all[idx_train].reshape(n_pts, -1))
  y_train = jnp.float32(y_all[idx_train].reshape(n_pts, -1))
  s_train = jnp.float32(s_all[idx_train].reshape(n_pts, -1))

  batch_key = random.fold_in(random.PRNGKey(cfg.get("seed", 42)), batch_idx)

  for alpha in cfg.alpha_list:
    for lr in cfg.lr_list:
      # lo and hi share the init key and minibatch stream, as in
      # burgersSCQR.py (both used PRNGKey(42)).
      init_key = random.fold_in(batch_key, 0)
      data_key = random.fold_in(batch_key, 1)

      for label, tau in (("lo", alpha / 2), ("hi", 1 - alpha / 2)):
        print(
            f"--- batch {batch_idx} ({batch_n_ens} members): "
            f"alpha={alpha} lr={lr} {label} (tau={tau}) ---"
        )
        dataset = DataGenerator(
            u_train, y_train, s_train, cfg.batch_size, rng_key=data_key
        )
        ensemble = build_ensemble(cfg, batch_n_ens, tau, lr, init_key)
        ensemble.train(BatchedDataGenerator(dataset, n_ens=batch_n_ens), nIter=cfg.nIter)

        save_batch(
            os.path.join(quantile_dir(workdir_root, alpha, lr, label), f"batch_{batch_idx}"),
            ensemble,
            meta={
                "n_ens": batch_n_ens,
                "tau": tau,
                "alpha": alpha,
                "lr": lr,
                "batch_idx": batch_idx,
            },
        )


if __name__ == "__main__":
  app.run(main)
